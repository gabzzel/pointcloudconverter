import numpy
import numpy as np
import pye57  # https://pypi.org/project/pye57/
import laspy  # https://pypi.org/project/laspy/
import plyfile  # https://pypi.org/project/plyfile/ and https://python-plyfile.readthedocs.io/en/latest/index.html
from plyfile import PlyData
from pye57 import e57


class PointCloud:
    def __init__(self):
        self.points_default_dtype = np.dtype(np.float32)
        self.points: np.ndarray = None

        self.intensities_default_dtype = np.dtype(np.float32)
        self.intensities: np.ndarray = None  # Default type is np.float32

        self.color_default_dtype = np.dtype(np.uint8)
        self.colors: np.ndarray = None  # Default type is np.uint8

    # https://pypi.org/project/pye57/
    def read_e57(self, filename: str):
        e57_object = pye57.E57(filename)
        data = e57_object.read_scan_raw(0)
        header: pye57.ScanHeader = e57_object.get_header(0)

        fm = self.map_field_names(header.point_fields)  # Field mapping

        self.points = np.stack((data[fm['x']], data[fm['y']], data[fm['z']]), axis=1) \
            if fm['x'] and fm['y'] and fm['z'] \
            else np.zeros(shape=(header.point_count, 3), dtype=self.points_default_dtype)

        self.intensities = data[fm['intensity']] \
            if fm['intensity'] \
            else np.zeros(shape=(header.point_count,), dtype=self.intensities_default_dtype)

        self.colors = np.stack((data[fm['r']], data[fm['g']], data[fm['b']]), axis=1) \
            if fm['r'] and fm['g'] and fm['b'] \
            else np.zeros(shape=(header.point_count, 3), dtype=self.color_default_dtype)

    # https://pypi.org/project/pye57/
    def write_e57(self, filename: str):
        e57 = pye57.E57(filename, mode="w")

        raw_data = dict()
        raw_data["cartesianX"] = self.points[:, 0]
        raw_data["cartesianY"] = self.points[:, 1]
        raw_data["cartesianZ"] = self.points[:, 2]

        raw_data["intensity"] = self.intensities

        # e57 expects the color data to be ints between 0 and 255 (incl.)

        # If we have float RGB data, we expect it to be in [0,1]
        if np.issubdtype(self.colors.dtype, np.floating):
            max_value = np.iinfo(np.uint8).max
            raw_data["colorRed"] = (self.colors[:, 0] * max_value).astype(np.uint8)
            raw_data["colorGreen"] = (self.colors[:, 1] * max_value).astype(np.uint8)
            raw_data["colorBlue"] = (self.colors[:, 2] * max_value).astype(np.uint8)

        e57.write_scan_raw(raw_data)

    # https://github.com/laspy/laspy/blob/740153c7b75abbea240d0b18a07f03038469f1fd/docs/complete_tutorial.rst#L76
    def read_las(self, filename: str):
        las = laspy.read(filename)

        has_rgb = 0
        intensities_name = None
        for dimension in las.point_format.dimensions:
            n = dimension.name.lower()
            if n == "red" or n == "r" or n == "green" or n == "g" or n == "blue" or n == "b":
                has_rgb += 1
            if n == "intensity" or n == "intensities":
                intensities_name = dimension.name

        # We can directly assign this, since the getter returns a new array
        self.points = las.xyz.astype(dtype=np.float32)

        if has_rgb == 3:
            r = np.array(las.red)
            g = np.array(las.green)
            b = np.array(las.blue)
            self.colors = np.stack((r, g, b), axis=1)
        else:
            self.colors = np.zeros_like(self.points, dtype=self.color_default_dtype)

        if intensities_name is not None:
            self.intensities = np.array(las[intensities_name])
        else:
            self.intensities = np.zeros(shape=(len(self.points),), dtype=self.intensities_default_dtype)

        x = 0

    def write_las(self, filename: str):
        point_format = 3
        header = laspy.LasHeader(point_format=point_format)

        header.offsets = np.min(self.points, axis=0)
        header.scales = np.max(self.points - header.offsets, axis=0) / np.iinfo(np.int32).max
        las = laspy.LasData(header=header)

        # TODO properly write the data. We cannot handle fractional values well.
        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        int16_max = np.iinfo(np.uint16).max

        # Intensity is always unsigned 16bit with las, meaning the max value is 65535
        las.intensity = self.intensities.astype(np.uint16)

        # TODO writing fix this based on type
        if self.colors is not None:
            # Like intensity, colors are unsigned 16bit, so we need to normalize to [0-65536]
            las.red = (self.colors[:, 0] * int16_max).astype(np.uint16)
            las.green = (self.colors[:, 1] * int16_max).astype(np.uint16)
            las.blue = (self.colors[:, 2] * int16_max).astype(np.uint16)

        las.write(filename)

    def read_ply(self, filename: str):
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

    # https://python-plyfile.readthedocs.io/en/latest/usage.html#creating-a-ply-file
    def write_ply(self, filename: str):

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

    def map_field_names(self, from_field_names: list[str]) -> dict:
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
