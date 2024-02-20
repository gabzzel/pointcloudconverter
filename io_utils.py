import os
from pathlib import Path
from typing import Optional

import numpy as np


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
