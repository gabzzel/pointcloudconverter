import logging
from argparse import ArgumentParser
import sys
from pathlib import Path
import time
import pye57

from PointCloud import PointCloud


def parse_args(valid_extensions, raw_args=None):
    args = sys.argv[1:] if raw_args is None else raw_args

    if len(args) < 1:
        print(f"No filepath provided. Cancelling.")
        return None

    file_path_raw = args[0].strip()

    read_path = Path(file_path_raw)
    if not read_path.exists():
        print(f"ERROR: File at path {read_path} does not exist. Cancelling.")
        return None

    read_extension = read_path.suffix.lower()

    if not (read_extension in valid_extensions):
        print(f"ERROR: File {read_path} has unsupported extension {read_extension}. Cancelling.")
        return None

    write_extension = ".las"
    write_path = Path(file_path_raw).with_suffix(write_extension)

    if len(args) > 1:
        destination = args[1].strip().lower()
        if destination.startswith('.') and destination in valid_extensions:
            write_extension = destination
        else:
            print(f"WARNING: Extension {destination} is not valid or permitted. "
                  f"Permitted extensions are {valid_extensions}. Using default .las")

    if not (write_extension in valid_extensions):
        print(f"WARNING: Provided write extension {write_extension} is not valid. Using default .las")

    if write_extension == read_extension:
        print(f"ERROR: The read file {read_path} has already the target extension {write_extension}. Cancelling.")
        return None

    write_path = write_path.with_suffix(write_extension)

    return read_path, read_extension, write_path, write_extension


def execute(raw_args):
    readers_dict = {
        ".las": "readers.las",
        ".ply": "readers.ply",
        ".e57": "readers.e57",
        ".pts": "readers.pts",
        ".pcd": "readers.pcd"
    }

    valid_extensions = list(readers_dict.keys())

    args = parse_args(valid_extensions, raw_args)
    if args is None:
        return

    read_path, read_extension, write_path, write_extension = args

    point_cloud = PointCloud()

    if read_extension == ".e57":
        point_cloud.read_e57(str(read_path))
    elif read_extension == ".las":
        point_cloud.read_las(str(read_path))

    if write_extension == ".e57":
        point_cloud.write_e57(str(write_path))

    print(args)





if __name__ == '__main__':
    if len(sys.argv) > 1:
        execute(sys.argv[1:])
    else:
        print("No arguments given.")
        time.sleep(10)
