import os
import tempfile
from typing import Optional
import re

import numpy as np
import pye57  # https://pypi.org/project/pye57/
import laspy  # https://pypi.org/project/laspy/
import plyfile  # https://pypi.org/project/plyfile/ and https://python-plyfile.readthedocs.io/en/latest/index.html
from plyfile import PlyData
import pypcd4  # https://pypi.org/project/pypcd4/
from tqdm import tqdm
import subprocess

import util
from custom_overwrites.CustomPly import CustomPlyElement
from custom_overwrites.LasDataOverwrite import CustomLasData
from io_utils import find_potreeconverter, convert_type_integers_incl_scaling, get_skip_lines_pts
from util import convert_to_type_incl_scaling, map_field_names


class PointCloud:
    def __init__(self):
        self.points_default_dtype = np.dtype(np.float32)
        self.points: np.ndarray = None
        # self.offsets: np.ndarray = None
        # self.scales: np.ndarray = None

        self.intensities_default_dtype = np.dtype(np.float32)
        self.intensities: np.ndarray = None  # Default type is np.float32

        self.color_default_dtype = np.dtype(np.uint8)
        self.colors: np.ndarray = None  # Default type is np.uint8

        self.custom_bounds = None

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def z(self):
        return self.points[:, 2]

    @property
    def r(self):
        return self.colors[:, 0]

    @property
    def g(self):
        return self.colors[:, 1]

    @property
    def b(self):
        return self.colors[:, 2]

    # https://pypi.org/project/pye57/
    def read_e57(self, filename: str) -> bool:
        e57_object = pye57.E57(filename)
        data = e57_object.read_scan(0, row_column=False, transform=True, intensity=True, colors=True,
                                    ignore_missing_fields=True)
        header: pye57.ScanHeader = e57_object.get_header(0)

        fm = map_field_names(header.point_fields)  # Field mapping

        self.points = np.stack((data[fm['x']], data[fm['y']], data[fm['z']]), axis=1) \
            if fm['x'] and fm['y'] and fm['z'] \
            else np.zeros(shape=(header.point_count, 3), dtype=self.points_default_dtype)

        self.intensities = data[fm['intensity']] if fm['intensity'] \
            else np.zeros(shape=(header.point_count,), dtype=self.intensities_default_dtype)

        # Probably the .e57 has colors in [0,255]
        if fm['r'] and fm['g'] and fm['b']:
            self.colors = np.stack((data[fm['r']], data[fm['g']], data[fm['b']]), axis=1)
        else:
            self.colors = np.zeros(shape=(header.point_count, 3), dtype=self.color_default_dtype)

        if header.cartesianBounds is not None:
            self.custom_bounds = np.array([header.xMinimum, header.xMaximum, header.yMinimum, header.yMaximum,
                                           header.zMinimum, header.zMaximum])

        return True

    # https://pypi.org/project/pye57/
    def write_e57(self, filename: str, verbose: int = 0) -> bool:
        e57_object = pye57.E57(filename, mode="w")

        raw_data = dict()
        raw_data["cartesianX"] = self.points[:, 0]
        raw_data["cartesianY"] = self.points[:, 1]
        raw_data["cartesianZ"] = self.points[:, 2]

        raw_data["intensity"] = self.intensities

        # e57 expects the color data to be ints between 0 and 255 (incl.)
        self.colors = convert_to_type_incl_scaling(self.colors, np.uint8, True)
        raw_data["colorRed"] = self.colors[:, 0]
        raw_data["colorGreen"] = self.colors[:, 1]
        raw_data["colorBlue"] = self.colors[:, 2]

        e57_object.write_scan_raw(raw_data)
        return True

    # https://github.com/laspy/laspy/blob/740153c7b75abbea240d0b18a07f03038469f1fd/docs/complete_tutorial.rst#L76
    def read_las(self, filename: str) -> bool:
        las = laspy.read(filename)
        fn = map_field_names([dimension.name for dimension in las.point_format.dimensions])

        # We can directly assign this, since the getter returns a new array
        # TODO get scales and offsets.
        self.points = las.xyz.astype(dtype=np.float32)

        self.colors = np.stack((np.array(las[fn['r']]), np.array(las[fn['g']]), np.array(las[fn['b']])), axis=1) \
            if 'r' in fn and 'g' in fn and 'b' in fn \
            else np.zeros_like(self.points, dtype=self.color_default_dtype)

        self.intensities = np.array(las[fn['intensity']]) \
            if 'intensity' in fn \
            else np.zeros(shape=(len(self.points),), dtype=self.intensities_default_dtype)

        return True

    def write_las(self, filename: str, verbose: int = 0) -> bool:
        point_format = 3
        header = laspy.LasHeader(point_format=point_format)

        if np.issubdtype(self.points.dtype, np.integer):
            header.offsets = np.array([0.0, 0.0, 0.0])
            header.scales = np.array([1.0, 1.0, 1.0])
            self.points = convert_type_integers_incl_scaling(self.points, np.int32)

        elif np.issubdtype(self.points.dtype, np.floating):
            header.offsets = np.min(self.points, axis=0)
            header.scales = np.max(self.points - header.offsets, axis=0) / np.iinfo(np.int32).max

        if self.custom_bounds is not None:
            header.maxs = np.array([self.custom_bounds[1], self.custom_bounds[3], self.custom_bounds[5]])
            header.mins = np.array([self.custom_bounds[0], self.custom_bounds[2], self.custom_bounds[4]])

        las = CustomLasData(header)
        max_allowed = np.iinfo(np.int32).max * header.scales + header.offsets
        min_allowed = np.iinfo(np.int32).min * header.scales + header.offsets

        las.x = np.clip(self.points[:, 0], min_allowed[0], max_allowed[0])
        las.y = np.clip(self.points[:, 1], min_allowed[1], max_allowed[1])
        las.z = np.clip(self.points[:, 2], min_allowed[2], max_allowed[2])

        # Intensity and color is always unsigned 16bit with las, meaning the max value is 65535
        las.intensity = convert_to_type_incl_scaling(self.intensities, np.dtype(np.uint16), True)

        self.colors = convert_to_type_incl_scaling(self.colors, np.dtype(np.uint16), True)
        las.red = self.colors[:, 0]
        las.green = self.colors[:, 1]
        las.blue = self.colors[:, 2]
        las.write(filename, verbose=verbose)

        return True

    def read_ply(self, filename: str) -> bool:
        ply: plyfile.PlyElement = PlyData.read(filename).elements[0]

        x = None
        y = None
        z = None
        r = None
        g = None
        b = None

        for prop in ply.properties:
            pn = prop.name.lower()

            if pn == 'x':
                x = ply[prop.name]
            elif pn == 'y':
                y = ply[prop.name]
            elif pn == 'z':
                z = ply[prop.name]
            elif pn == 'r' or pn == 'red':
                r = ply[prop.name]
            elif pn == 'g' or pn == 'green':
                g = ply[prop.name]
            elif pn == 'b' or pn == 'blue':
                b = ply[prop.name]
            elif 'intensity' in pn or "intensities" in pn:
                self.intensities = np.array(ply[prop.name])

        self.points = np.stack((x, y, z), axis=1)

        if r is not None and g is not None and b is not None:
            self.colors = np.stack((r, g, b), axis=1)
        else:
            self.colors = np.zeros_like(self.points, dtype=np.uint8)

        if self.intensities is None:
            self.intensities = np.zeros(shape=(len(self.points),), dtype=np.float32)
        return True

    # https://python-plyfile.readthedocs.io/en/latest/usage.html#creating-a-ply-file
    def write_ply(self, filename: str, verbose: int = 0) -> bool:

        point_type = self.points.dtype.name
        color_type = self.colors.dtype.name
        intensity_type = self.intensities.dtype.name

        properties = [
            plyfile.PlyProperty('x', point_type),
            plyfile.PlyProperty('y', point_type),
            plyfile.PlyProperty('z', point_type),
            plyfile.PlyProperty('red', color_type),
            plyfile.PlyProperty('green', color_type),
            plyfile.PlyProperty('blue', color_type),
            plyfile.PlyProperty('scalar_Intensity', intensity_type)
        ]

        el = CustomPlyElement(name="points", properties=properties, count=str(len(self.points)), comments=[],
                              verbose=verbose)
        dtype_list = [('x', point_type), ('y', point_type), ('z', point_type),
                      ('red', color_type), ('green', color_type), ('blue', color_type),
                      ('scalar_Intensity', intensity_type)]

        data = np.empty(shape=(len(self.points),), dtype=dtype_list)
        data['x'] = self.points[:, 0]
        data['y'] = self.points[:, 1]
        data['z'] = self.points[:, 2]
        data['red'] = self.colors[:, 0]
        data['green'] = self.colors[:, 1]
        data['blue'] = self.colors[:, 2]
        data['scalar_Intensity'] = self.intensities
        el.data = data
        PlyData([el]).write(filename)
        return True

    # http://www.paulbourke.net/dataformats/pts/
    def read_pts(self, filename: str) -> bool:

        dtype_list = [
            ('x', np.dtype(np.float32)), ('y', np.dtype(np.float32)), ('z', np.dtype(np.float32)),
            ('intensity', np.dtype(np.float32)),
            ('r', np.dtype(np.uint8)), ('g', np.dtype(np.uint8)), ('b', np.dtype(np.uint8))
        ]

        skip_lines = get_skip_lines_pts(filename)
        with open(filename, mode='rb') as f:

            # Skip the header and comments
            for i in range(skip_lines):
                f.readline()

            # Read a sample and extract the required types
            sample = f.readline().split(b' ')
            for index, sample_element in enumerate(sample):
                sample_element = sample_element.strip()
                n = dtype_list[index][0]
                dtype_list[index] = (n, np.dtype(np.uint8)) if sample_element.isdigit() else (n, np.dtype(np.float32))

        data = np.loadtxt(fname=filename, dtype=dtype_list, comments="//", delimiter=' ', skiprows=skip_lines)
        self.points = np.stack((data['x'], data['y'], data['z']), axis=1)
        self.intensities = data['intensity']
        self.colors = np.stack((data['r'], data['g'], data['b']), axis=1)
        return True

    def write_pts(self, filename: str, verbose: int = 0) -> bool:
        # Make sure the colors are the correct format.
        if np.issubdtype(self.colors.dtype, np.integer) and self.colors.dtype != np.dtype(np.uint8):
            self.colors = convert_type_integers_incl_scaling(self.colors, np.dtype(np.uint8))
        elif np.issubdtype(self.colors.dtype, np.floating) and self.colors.dtype != np.dtype(np.float32):
            self.colors = self.colors.astype(np.float32)

        self.intensities = convert_to_type_incl_scaling(self.intensities, np.dtype(np.float32), True)

        float_addition = 'f' if np.issubdtype(self.colors.dtype, np.floating) else ''
        header = f"X Y Z Intensity R{float_addition} G{float_addition} B{float_addition}\n"

        with open(filename, mode='w') as f:
            f.write(header)
            f.write(f"{len(self.points)}\n")

            for i in tqdm(range(len(self.points)), unit="points", leave=True, desc="Writing .pts file..."):
                f.write(
                    " ".join((str(self.points[i][0]), str(self.points[i][1]), str(self.points[i][2]),
                              str(self.intensities[i]),
                              str(self.colors[i][0]), str(self.colors[i][1]), str(self.colors[i][2])))
                )
                f.write("\n")
        return True

    def write_potree(self,
                     current_file: str,
                     target_directory: str,
                     potreeconverter_path: Optional[str] = None,
                     verbosity: int = 0):
        # Find the Potree converter by
        # (1) the given path if the given path is a file path,
        # (2) inside the given directory if the given path is a directory
        # (3) inside the folder of the currently executed file or any of its subdirectories.
        potree_exe = find_potreeconverter(current_file, potreeconverter_path)
        if potree_exe is None:
            if verbosity == 1 or verbosity == 2:
                print(f"Could not find potreeconverter. Conversion cancelled.")
            return False

        if verbosity == 1 or verbosity == 2:
            print(f"Found potree converter at {potree_exe}.")

        tempdir = tempfile.gettempdir()
        temp_las_file = str(os.path.join(tempdir, "templas.las"))

        self.write_las(temp_las_file, verbose=verbosity)  # Create a temporary las file.

        #subprocess.run([str(potree_exe), temp_las_file, "-o", target_directory], stdout=subprocess.DEVNULL,
        #               stderr=subprocess.STDOUT)

        arguments = [temp_las_file, '-o', target_directory]
        command = [potree_exe] + arguments
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        if verbosity == 2 or verbosity == 3:

            progress_bar = tqdm(desc="Writing potree", total=100, unit='%') if verbosity == 2 else None
            percent = 0
            for stdout_line in iter(process.stdout.readline, ''):
                try:
                    first_percent_index = stdout_line.index('%')
                    new_percent = int(re.findall(r'\d+', stdout_line[:first_percent_index])[0])
                    increase = max(0, new_percent - percent)
                    percent = new_percent
                    if progress_bar:
                        progress_bar.update(increase)
                    else:
                        print(f"w{percent}")

                except ValueError:
                    continue

        # If we want to print errors.
        # for stderr_line in iter(process.stderr.readline, ''):
        #    print("Standard Error:", stderr_line, end='')

        process.communicate()

        if verbosity == 3:  # Print that we are done.
            print("w100")

        os.remove(temp_las_file)  # Clean up
        return True

    def read_pcd(self, filename: str):
        pc: pypcd4.PointCloud = pypcd4.PointCloud.from_path(filename)
        fn = util.map_field_names(pc.fields)

        if fn['x'] is not None and fn['y'] is not None and fn['z'] is not None:
            x = np.squeeze(pc.numpy(fields=fn['x']))
            y = np.squeeze(pc.numpy(fields=fn['y']))
            z = np.squeeze(pc.numpy(fields=fn['z']))
            self.points = np.stack((x, y, z), axis=1)
        else:
            self.points = np.zeros(shape=(pc.points, 3), dtype=np.float32)

        if fn['r'] is not None and fn['g'] is not None and fn['b'] is not None:
            r = np.squeeze(pc.numpy(fields=fn['r']))
            g = np.squeeze(pc.numpy(fields=fn['g']))
            b = np.squeeze(pc.numpy(fields=fn['b']))
            self.colors = np.stack((r, g, b), axis=1)
        elif 'rgb' in pc.fields:
            self.colors = pc.decode_rgb(pc.numpy(fields=['rgb']))
        else:
            self.colors = np.zeros_like(self.points, dtype=self.color_default_dtype)

        if fn['intensity'] is not None:
            self.intensities = np.squeeze(pc.numpy(fields=[fn['intensity']]))
        else:
            self.intensities = np.zeros(shape=(len(self.points), ), dtype=self.intensities_default_dtype)

    # https://pypi.org/project/pypcd4/
    def write_pcd(self, filename: str, verbose: int = 0) -> bool:
        fields = ['x', 'y', 'z', 'intensity', 'rgb']
        types = [self.points.dtype, self.points.dtype, self.points.dtype, self.intensities.dtype]

        # RGB conversion creates a 1D np.float32 array out of a np.uint8 array.
        rgb = pypcd4.PointCloud.encode_rgb(convert_to_type_incl_scaling(self.colors, np.dtype(np.uint8), True))
        types.append(rgb.dtype)

        try:
            pc = pypcd4.PointCloud.from_points(points=[self.x, self.y, self.z, self.intensities, rgb],
                                               fields=fields,
                                               types=types)
            pc.save(fp=filename, encoding=pypcd4.Encoding.BINARY)
            return True

        except Exception as e:
            print(f"Error: could not write .pcd: {e}")

        return False

