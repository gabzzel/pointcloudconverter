import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path
import argparse

import PointCloud
import io_utils
from typing import Tuple, Optional


def parse_args(valid_extensions) -> Optional[Tuple[Path, str, Path, str]]:
    description = "This program is used to convert point clouds of different formats from and to each other."
    parser = argparse.ArgumentParser(prog="Point Cloud Converter", description=description)
    parser.add_argument('origin_path',
                        help='Path to the point cloud file to convert.',
                        action='store',
                        type=Path)

    parser.add_argument('-destination_path', '-destination', '-dest', '-d',
                        help='Path of the destination of the converted point cloud. If a folder, places converted '
                             'pointcloud in that folder. If a file path, writes the converted point cloud to that '
                             'file path.',
                        action='store',
                        type=Path)

    parser.add_argument('-extension', '-ext', '-e',
                        help='The target extension or format of the converted point cloud. '
                             'Ignored when the destination_path is a file path. Defaults to .las',
                        choices=valid_extensions,
                        default='.las',
                        type=str,
                        action='store')

    parser.add_argument('-unsafe', '-u', '-overwrite',
                        help='Whether to allow overwriting of existing point cloud.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    origin_path: Path = args.origin_path

    if not origin_path.is_file():
        print(f"ERROR: The provided origin path is not a file ({origin_path}). Cancelling.")
        return None

    if origin_path.suffix not in valid_extensions:
        print(f"ERROR: The provided file does not have a valid extension. Found {origin_path.suffix} but expected "
              f"one of {valid_extensions}. Cancelling.")
        return None

    print(f"SUCCESS: Found valid point cloud file at {origin_path}")

    origin_file_name = origin_path.stem
    origin_directory: Path = origin_path.parent
    provided_destination_path: Path = args.destination_path
    destination_path = None

    # Case 1: Destination path and extension are not provided.
    if args.destination_path is None and args.extension is None:
        print(f"WARNING: No destination path of extension provided. Defaulting to extension .las. "
              f"The converted point cloud will be stored in the same folder as the origin: ({origin_directory})")
        destination_path = origin_directory.joinpath(origin_file_name, '.las')

    # Case 2: destination path is a (valid) file. Ignore the extension.
    elif (args.destination_path is not None and provided_destination_path.is_file() and
          provided_destination_path.suffix in valid_extensions):
        print(f"Found valid destination file: {provided_destination_path}")
        destination_path = provided_destination_path

        if args.extension is not None:
            print(f"WARNING: A valid extension/format ({args.extension}) is given, but will be ignored since "
                  f"the destination path already provides the target extension {destination_path.suffix}")

    # Case 3: destination path is a directory and extension is available
    elif args.destination_path is not None and provided_destination_path.is_dir() and args.extension is not None:
        print(f"SUCCESS: Found valid destination directory and extension.")
        destination_path = provided_destination_path.joinpath(origin_file_name + args.extension)

    # Case 4: destination path is directory and extension is not available.
    elif args.destination_path is not None and provided_destination_path.is_dir() and args.extension is None:
        print(f"SUCCESS: A valid destination directory has been found but no extension. Default .las will be used.")
        destination_path = provided_destination_path

    # Case 5: destination path is not provided and extension is provided.
    elif args.destination_path is None and args.extension is not None:
        print(f"SUCCESS: Valid extension {args.extension} has been provided. Point cloud will be placed in the same"
              f" directory as the original point cloud.")
        destination_path = origin_directory.joinpath(origin_file_name + args.extension)

    if destination_path is None:
        print(f"ERROR: Could not parse arguments. Cancelling.")
        return None

    # If the format should be potree, the destination should be a folder, not a file.
    if args.extension == "potree":
        destination_path = destination_path.parent.joinpath(origin_file_name + "_potree")

    if not args.unsafe and destination_path.exists():
        print("ERROR: Provided safe execution (i.e. no file or folder overwriting) and found existing file or folder "
              f"{destination_path}. Exitting.")
        return None

    return origin_path, origin_path.suffix, destination_path, args.extension


def execute(raw_args):
    extensions = [".las", ".ply", ".e57", ".pts", ".pcd", "potree"]
    args = parse_args(extensions)

    if args is None:
        return

    read_path, read_extension, write_path, write_extension = args

    point_cloud = PointCloud.PointCloud()

    readers = {
        ".las": point_cloud.read_las,
        ".laz": point_cloud.read_las,
        ".ply": point_cloud.read_ply,
        ".e57": point_cloud.read_e57,
        ".pts": point_cloud.read_pts,
        ".pcd": point_cloud.read_pcd
    }
    writers = {
        ".las": point_cloud.write_las,
        ".ply": point_cloud.write_ply,
        ".e57": point_cloud.write_e57,
        ".pts": point_cloud.write_pts,
        ".pcd": point_cloud.write_pcd
    }

    # Special case where we can directly call the Potree converter
    if (read_extension == ".las" or read_extension == ".laz") and write_extension == "potree":
        ptc = io_utils.find_potreeconverter(sys.argv[0])
        if ptc is None:
            print(f"Could not find potree converter! Cancelling.")
            return
        subprocess.run([str(ptc), read_path, "-o", write_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Used potree converter at {ptc} for convert {read_path} to potree.")
        return

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
        start_time = time.time()
        execute(sys.argv[1:])
        elapsed = round(time.time() - start_time, 3)
        print("Finished in ", elapsed, " seconds.")
    else:
        print("No arguments given.")
        time.sleep(5)
