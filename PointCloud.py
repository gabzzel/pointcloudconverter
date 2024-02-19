import sys

import numpy
import numpy as np
import pye57  # https://pypi.org/project/pye57/
import laspy  # https://pypi.org/project/laspy/
import plyfile  # https://pypi.org/project/plyfile/ and https://python-plyfile.readthedocs.io/en/latest/index.html
from plyfile import PlyData
from pye57 import e57

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
        return (as_float64 / max_value_for_type(array.dtype) * np.finfo(target_type)).astype(target_type)


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

        x = 0

    # https://pypi.org/project/pye57/
    def write_e57(self, filename: str):
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

        if np.issubdtype(self.points.dtype, np.integer):
            header.offsets = np.array([0.0, 0.0, 0.0])
            header.scales = np.array([1.0, 1.0, 1.0])
            self.points = convert_type_integers_incl_scaling(self.points, np.int32)

        elif np.issubdtype(self.points.dtype, np.floating):
            # TODO, this does not work!
            header.offsets = np.min(self.points, axis=0)
            header.scales = np.max(self.points - header.offsets, axis=0) / np.iinfo(np.int32).max
            scaling_factor = np.iinfo(np.int32).max / max_value_for_type(self.points.dtype)
            self.points = (self.points * scaling_factor).astype(np.int32)

        las = laspy.LasData(header)
        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        # Intensity and color is always unsigned 16bit with las, meaning the max value is 65535
        las.intensity = convert_to_type_incl_scaling(self.intensities, np.dtype(np.uint16), False)

        self.colors = convert_to_type_incl_scaling(self.colors, np.dtype(np.uint16), True)
        las.red = self.colors[:, 0]
        las.green = self.colors[:, 1]
        las.blue = self.colors[:, 2]
        las.header = header
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

