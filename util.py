import sys
from typing import Union, Tuple, List

import numpy as np

from io_utils import convert_type_integers_incl_scaling


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


def map_field_names(from_field_names: Union[List[str], Tuple[str]]) -> dict:
    mapping = {'x': None, 'y': None, 'z': None, 'r': None, 'g': None, 'b': None, 'intensity': None}

    for field_name in from_field_names:
        field_name_lowered: str = field_name.lower()

        # Get both 'intensity' and 'intensities'
        if mapping['intensity'] is None and "intensit" in field_name_lowered:
            mapping['intensity'] = field_name
        elif mapping['x'] is None and "x" in field_name_lowered:
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

    return mapping


def convert_type_floats_incl_scaling(array, target_type):
    current_max_value = np.finfo(array.dtype).max
    target_max_value = np.finfo(target_type).max
    # First divide by the max value and then multiply by new max value, not the other way around to prevent overflow!
    return (array / current_max_value) * target_max_value.astype(dtype=target_type)
