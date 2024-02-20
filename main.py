import logging
import os
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
    write_path = Path(file_path_raw).with_suffix(write_extension) \
        if args[1].strip().lower() != "potree" \
        else Path(str(os.path.join(read_path.parent, read_path.stem + "_potree")))

    if len(args) > 1:
        destination = args[1]
        if destination.strip().lower() in valid_extensions:
            write_extension = destination
        else:
            print(f"WARNING: Extension {destination} is not valid or permitted. "
                  f"Permitted extensions are {valid_extensions}. Using default .las")

    if not (write_extension in valid_extensions):
        print(f"WARNING: Provided write extension {write_extension} is not valid. Using default .las")

    if write_extension == read_extension:
        print(f"ERROR: The read file {read_path} has already the target extension {write_extension}. Cancelling.")
        return None

    if write_extension != "potree":
        write_path = write_path.with_suffix(write_extension)

    return read_path, read_extension, write_path, write_extension


def execute(raw_args):

    extensions = [".las", ".ply", ".e57", ".pts", ".pcd", "potree"]
    args = parse_args(extensions, raw_args)
    if args is None:
        return

    read_path, read_extension, write_path, write_extension = args

    point_cloud = PointCloud()

    readers = {
        ".las": point_cloud.read_las,
        ".laz": point_cloud.read_las,
        ".ply": point_cloud.read_ply,
        ".e57": point_cloud.read_e57,
        ".pts": point_cloud.read_pts
        # ".pcd": "readers.pcd"
    }
    writers = {
        ".las": point_cloud.write_las,
        ".ply": point_cloud.write_ply,
        ".e57": point_cloud.write_e57,
        ".pts": point_cloud.write_pts,
    }

    success = False
    if read_extension in readers:
        start_time = time.time()
        success = readers[read_extension](str(read_path))
        elapsed = round(time.time() - start_time, 3)
        if success:
            print(f"Successfully read file {read_path} [{elapsed}s]")

    if not success:
        print(f"Could not read file at {read_path}. Cancelling.")
        time.sleep(5)
        return

    if write_extension in writers:
        start_time = time.time()
        success = writers[write_extension](str(write_path))
        elapsed = round(time.time() - start_time, 3)
        if success:
            print(f"Written file {write_path} [{elapsed}s]")
    elif write_extension == "potree":
        start_time = time.time()
        success = point_cloud.write_potree(current_file=sys.argv[0], target_directory=str(write_path))
        elapsed = round(time.time() - start_time, 3)
        if success:
            print(f"Written file {write_path} [{elapsed}s]")

    if not success:
        print(f"Could not write file at {write_path}.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        execute(sys.argv[1:])
    else:
        print("No arguments given.")
        time.sleep(5)
