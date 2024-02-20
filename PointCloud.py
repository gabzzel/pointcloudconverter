import os
import sys
import tempfile
from typing import Optional, Any

import numpy as np
import pye57  # https://pypi.org/project/pye57/
import laspy  # https://pypi.org/project/laspy/
import plyfile  # https://pypi.org/project/plyfile/ and https://python-plyfile.readthedocs.io/en/latest/index.html
from plyfile import PlyData
from tqdm import tqdm
from pathlib import Path
import subprocess


def max_value_for_type(data_type):
    if np.issubdtype(data_type, np.integer):
        if isinstance(data_type, np.dtype):
            return np.iinfo(data_type).max
        elif data_type == int:
            return sys.maxsize
    elif np.issubdtype(data_type, np.floating):
        if isinstance(data_type, np.dtype):
            return np.finfo(data_type).max
        elif data_type == float:
            return sys.float_info.max
    else:
        raise ValueError("Unsupported data type")


def convert_to_type_incl_scaling(array: np.ndarray, target_type: np.dtype, float_max_is_1: bool):
    """ If max value is None, the max value is inferred from the types. Else, this max value is used."""

    # We don't have to do anything
    if array.dtype == target_type:
        return array

    if np.issubdtype(array.dtype, np.floating) and np.issubdtype(target_type, np.floating):
        return convert_type_floats_incl_scaling(array, target_type)

    elif np.issubdtype(array.dtype, np.integer) and np.issubdtype(target_type, np.integer):
        return convert_type_integers_incl_scaling(array, target_type)

    # If we are currently dealing with a float, but we need to convert to integer...
    elif np.issubdtype(array.dtype, np.floating) and np.issubdtype(target_type, np.integer):
        # If we can assume the values are between 0 and 1, we can just multiply by the max value and return.
        if float_max_is_1:
            return (array * max_value_for_type(target_type)).astype(dtype=target_type)
        # If we cannot assume the values are between 0 and 1, we have to make sure we divide by the max first.
        else:
            max_value = max_value_for_type(array.dtype)
            return (array / max_value * max_value_for_type(target_type)).astype(dtype=target_type)

    # We are dealing with an integer that needs to be converted to a float.
    elif np.issubdtype(array.dtype, np.integer) and np.issubdtype(target_type, np.floating):
        as_float64 = array.astype(dtype=np.float64)  # Make sure we can handle the dividing
        scaling = 1.0 if float_max_is_1 else np.finfo(target_type).max
        return (as_float64 / max_value_for_type(array.dtype) * scaling).astype(target_type)


def convert_type_integers_incl_scaling(array, target_type):
    # We are dealing with a signed integer array to unsigned one.
    if np.iinfo(array.dtype).min < 0 and np.iinfo(target_type).min == 0:
        print(f"Warning! The original array possibly contains negative values. The signs will not be preserved "
              f"when converting to the new target type {target_type}.")

    current_max = np.iinfo(array.dtype).max + 1
    target_max = np.iinfo(target_type).max + 1
    if current_max > target_max:  # If we need to scale down, make sure we divide by a whole number.
        return (array / (current_max / target_max)).astype(dtype=target_type)
    else:  # If we need to scale up, target is higher than current, so we increase
        return (array * (target_max / current_max)).astype(dtype=target_type)


def convert_type_floats_incl_scaling(array, target_type):
    current_max_value = np.finfo(array.dtype).max
    target_max_value = np.finfo(target_type).max
    # First divide by the max value and then multiply by new max value, not the other way around to prevent overflow!
    return (array / current_max_value) * target_max_value.astype(dtype=target_type)


def map_field_names(from_field_names: list[str]) -> dict:
    mapping = {'x': None, 'y': None, 'z': None, 'r': None, 'g': None, 'b': None, 'intensity': None}

    for field_name in from_field_names:
        field_name_lowered: str = field_name.lower()
        if mapping['x'] is None and "x" in field_name_lowered:
            mapping['x'] = field_name
        elif mapping['y'] is None and "y" in field_name_lowered:
            mapping['y'] = field_name
        elif mapping['z'] is None and "z" in field_name_lowered:
            mapping['z'] = field_name
        elif mapping['r'] is None and (field_name_lowered == "r" or "red" in field_name_lowered):
            mapping['r'] = field_name
        elif mapping['g'] is None and (field_name_lowered == "g" or "green" in field_name_lowered):
            mapping['g'] = field_name
        elif mapping['b'] is None and (field_name_lowered == "b" or "blue" in field_name_lowered):
            mapping['b'] = field_name

        # Get both 'intensity' and 'intensities'
        elif mapping['intensity'] is None and "intensit" in field_name_lowered:
            mapping['intensity'] = field_name

    return mapping


def get_skip_lines_pts(filename: str):
    to_skip = 0
    with open(filename, 'rb') as f:
        for line in f:
            candidate_header = line.strip().lower()
            # We have encountered a header line containing the names of the columns,
            # or the line indicating the number of points
            if candidate_header.startswith(b"//") or candidate_header.isdigit():
                to_skip += 1
            # If we encounter a space in a line, and it's not a comment, we can continue reading the file.
            elif b' ' in candidate_header:
                return to_skip
    return to_skip


def find_file_in_directory(file_name, directory):
    p = directory if type(directory) is Path else Path(directory)
    if not p.exists() or not p.is_dir():
        return None

    # Walk through all files. If we find one that (when lowered) equals what we are looking for, return it!
    for (root, dirs, files) in os.walk(Path(directory)):
        files_lowered = [Path(str(os.path.join(root, file_path))).name.lower() for file_path in files]
        for index, file_name_lowered in enumerate(files_lowered):
            if file_name_lowered == file_name.lower():
                return Path(str(os.path.join(root, files[index])))
    return None


def find_potreeconverter(current_file: str, ptc_path: Optional[str] = None):
    # If a Potree Converter executable path is specified...
    if ptc_path is not None:
        p = Path(ptc_path)
        if not p.exists():  # If this path does not exist, just do nothing.
            print(f"Path {ptc_path} is not valid.")
            return None
        elif p.is_dir():
            potreeconverter_path = find_file_in_directory("potreeconverter.exe", p)
            if potreeconverter_path is None:
                print(f"Could not find potreeconverter in {p} or any of its subdirectories.")
            else:
                return potreeconverter_path
        elif p.is_file() and p.name.lower() == "potreeconverter.exe":
            return p

    # We did not get a Potree converter path specified. We need to look ourselves.
    else:
        p = Path(current_file)
        if not p.exists() or not p.is_file():
            print(f"The given file {p} is invalid!")
            return None
        potree_path = find_file_in_directory("potreeconverter.exe", p.parent)
        if potree_path is None:
            print(f"Could not find potreeconverter in {p.parent} or any of its subdirectories.")
            return None
        return potree_path
    return None


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

    # https://pypi.org/project/pye57/
    def read_e57(self, filename: str) -> bool:
        e57_object = pye57.E57(filename)
        data = e57_object.read_scan_raw(0)
        header: pye57.ScanHeader = e57_object.get_header(0)

        fm = map_field_names(header.point_fields)  # Field mapping

        self.points = np.stack((data[fm['x']], data[fm['y']], data[fm['z']]), axis=1, dtype=self.points_default_dtype) \
            if fm['x'] and fm['y'] and fm['z'] \
            else np.zeros(shape=(header.point_count, 3), dtype=self.points_default_dtype)

        self.intensities = data[fm['intensity']].astype(np.float32) \
            if fm['intensity'] \
            else np.zeros(shape=(header.point_count,), dtype=self.intensities_default_dtype)

        # We assume the .e57 has colors in [0,255]
        self.colors = np.stack((data[fm['r']], data[fm['g']], data[fm['b']]), axis=1, dtype=np.uint8) \
            if fm['r'] and fm['g'] and fm['b'] \
            else np.zeros(shape=(header.point_count, 3), dtype=self.color_default_dtype)

        return True

    # https://pypi.org/project/pye57/
    def write_e57(self, filename: str) -> bool:
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

    def write_las(self, filename: str) -> bool:
        point_format = 3
        header = laspy.LasHeader(point_format=point_format)

        if np.issubdtype(self.points.dtype, np.integer):
            header.offsets = np.array([0.0, 0.0, 0.0])
            header.scales = np.array([1.0, 1.0, 1.0])
            self.points = convert_type_integers_incl_scaling(self.points, np.int32)

        elif np.issubdtype(self.points.dtype, np.floating):
            header.offsets = np.min(self.points, axis=0)
            header.scales = np.max(self.points - header.offsets, axis=0) / np.iinfo(np.int32).max

        las = laspy.LasData(header)
        las.xyz = self.points

        # Intensity and color is always unsigned 16bit with las, meaning the max value is 65535
        las.intensity = convert_to_type_incl_scaling(self.intensities, np.dtype(np.uint16), False)

        self.colors = convert_to_type_incl_scaling(self.colors, np.dtype(np.uint16), True)
        las.red = self.colors[:, 0]
        las.green = self.colors[:, 1]
        las.blue = self.colors[:, 2]
        las.write(filename)

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
    def write_ply(self, filename: str) -> bool:

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

        el = plyfile.PlyElement(name="points", properties=properties, count=str(len(self.points)))
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

    def write_pts(self, filename: str) -> bool:
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

    def write_potree(self, current_file: str, target_directory: str, potreeconverter_path: Optional[str] = None):
        # Find the Potree converter by
        # (1) the given path if the given path is a file path,
        # (2) inside the given directory if the given path is a directory
        # (3) inside the folder of the currently executed file or any of its subdirectories.
        potree_exe = find_potreeconverter(current_file, potreeconverter_path)
        if potree_exe is None:
            print(f"Could not find potreeconverter. Conversion cancelled.")
            return False

        print(f"Found potree converter at {potree_exe}.")
        tempdir = tempfile.gettempdir()
        temp_las_file = str(os.path.join(tempdir, "templas.las"))
        self.write_las(temp_las_file)
        subprocess.run([str(potree_exe), temp_las_file, "-o", target_directory])
        os.remove(temp_las_file)
        return True
