import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import psutil
from colorama import Fore
from colorama import Style
from colorama import init as colorama_init

import PointCloud
import io_utils


def parse_args(valid_extensions) -> Optional[Tuple[Path, str, Path, str, bool]]:
    description = "This program is used to convert point clouds of different formats from and to each other."
    parser = argparse.ArgumentParser(prog="Point Cloud Converter", description=description)
    parser.add_argument('origin_path',
                        help='Path to the point cloud file to convert.',
                        action='store',
                        type=Path)

    parser.add_argument('-destination_path', '-destination', '-dest', '-d',
                        help='Path of the destination of the converted point cloud. If a folder, places converted '
                             'point cloud in that folder. If a file path, writes the converted point cloud to that '
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

    parser.add_argument('-verbose', '-v',
                        help="The verbosity level. Default 0 is nothing. 1 provides basic information. "
                             "2 prints the information pretty, with colors and progress bars. "
                             "3 prints the read and write progress percentages raw.",
                        action='store',
                        type=int,
                        choices=[0, 1, 2, 3],
                        default=0)

    args = parser.parse_args()
    origin_path: Path = args.origin_path
    verbose = args.verbose

    if not origin_path.is_file():
        log(f"The provided origin path is not a file ({origin_path}). Cancelling.", 'e', verbose)
        return None

    if origin_path.suffix not in valid_extensions:
        log(f"The provided file does not have a valid extension. Found {origin_path.suffix} but expected "
            f"one of {valid_extensions}. Cancelling.", 'e', verbose)
        return None

    log(f"Found valid point cloud file at {origin_path}", 's', verbose)

    origin_file_name = origin_path.stem
    origin_directory: Path = origin_path.parent
    provided_destination_path: Path = args.destination_path
    destination_path = None

    # Case 1: Destination path and extension are not provided.
    if args.destination_path is None and args.extension is None:
        log(f"No destination path of extension provided. Defaulting to extension .las. "
            f"The converted point cloud will be stored in the same folder as the origin: ({origin_directory})",
            'w', verbose)
        destination_path = origin_directory.joinpath(origin_file_name, '.las')

    # Case 2: destination path is a (valid) file. Ignore the extension.
    elif (args.destination_path is not None and provided_destination_path.is_file() and
          provided_destination_path.suffix in valid_extensions):
        log(f"Found valid destination file: {provided_destination_path}", 's', verbose)
        destination_path = provided_destination_path

        if args.extension is not None:
            log(f"A valid extension/format ({args.extension}) is given, but will be ignored since "
                f"the destination path already provides the target extension {destination_path.suffix}",
                'w', verbose)

    # Case 3: destination path is a directory and extension is available
    elif args.destination_path is not None and provided_destination_path.is_dir() and args.extension is not None:
        log(f"Found valid destination directory and extension.", 's', verbose)
        destination_path = provided_destination_path.joinpath(origin_file_name + args.extension)

    # Case 4: destination path is directory and extension is not available.
    elif args.destination_path is not None and provided_destination_path.is_dir() and args.extension is None:
        log(f"A valid destination directory has been found but no extension. Default .las will be used.",
            'w', verbose)
        destination_path = provided_destination_path

    # Case 5: destination path is not provided and extension is provided.
    elif args.destination_path is None and args.extension is not None:
        log(f"Valid extension {args.extension} has been provided. Point cloud will be placed in the same"
            f" directory as the original point cloud.", 'w', verbose)
        destination_path = origin_directory.joinpath(origin_file_name + args.extension)

    if destination_path is None:
        log(f"Could not parse arguments. Cancelling.", 'e', verbose)
        return None

    # If the format should be potree, the destination should be a folder, not a file.
    if args.extension == "potree":
        destination_path = destination_path.parent.joinpath(origin_file_name + "_potree")

    if not args.unsafe and destination_path.exists():
        log("Provided safe execution (i.e. no file or folder overwriting) and found existing file or folder "
            f"{destination_path}. Exiting.", 'e', verbose)
        return None

    return origin_path, origin_path.suffix, destination_path, args.extension, args.verbose


def execute():
    extensions = [".las", ".ply", ".e57", ".pts", ".pcd", "potree"]
    args = parse_args(extensions)

    if args is None:
        log("Could not parse arguments. Exiting.", 'e', True)
        time.sleep(5)
        return

    read_path, read_extension, write_path, write_extension, verbose = args

    if not check_file(read_path, verbose):
        log("File checking failed. Exiting.", 'e', verbose)
        return

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

    # Special case where we can directly use the potree converter
    if read_extension == ".las" and write_extension == "potree":
        point_cloud.write_potree(current_file=sys.argv[0],
                                 target_directory=str(write_path),
                                 verbosity=verbose,
                                 origin_las_path=str(read_path))
        log(f"Used potree converter for convert {read_path} to potree.", 's', verbose)
        return

    success = False
    if read_extension in readers:
        start_time = time.time()
        success = readers[read_extension](str(read_path))
        elapsed = round(time.time() - start_time, 3)
        if success:
            log(f"Successfully read file {read_path} [{elapsed}s]", 's', verbose)

    if not success:
        log(f"Could not read file at {read_path}. Cancelling.", 'e', verbose)
        time.sleep(5)
        return

    if write_extension in writers:
        start_time = time.time()
        success = writers[write_extension](str(write_path), verbose)
        elapsed = round(time.time() - start_time, 3)
        if success:
            log(f"Written file {write_path} [{elapsed}s]", 's', verbose)
    elif write_extension == "potree":
        start_time = time.time()
        success = point_cloud.write_potree(current_file=sys.argv[0], target_directory=str(write_path),
                                           verbosity=verbose, origin_las_path=None)
        elapsed = round(time.time() - start_time, 3)
        if success:
            log(f"Written point cloud at {write_path} [{elapsed}s]", 's', verbose)

    if not success:
        log(f"Could not write file at {write_path}.", 'e', verbose)


def log(msg: str, level: str, verbosity: int) -> None:
    if verbosity <= 0 or verbosity >= 3:
        return

    if verbosity == 1:
        print(msg)
        return

    level = level.lower().strip()
    if level == 'success' or level == 's':
        print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}:", msg)
    elif level == 'warning' or level == 'w':
        print(f"{Fore.YELLOW}WARNING{Style.RESET_ALL}:", msg)
    elif level == 'error' or level == 'er' or level == 'e':
        print(f"{Fore.RED}ERROR{Style.RESET_ALL}:", msg)


def check_file(file_path: os.PathLike, verbose: bool) -> bool:
    p = file_path
    if not isinstance(file_path, Path):
        p = Path(file_path)

    if not p.exists():
        log(f"File does not exist at {p}", 'e', verbose)
        return False

    if not p.is_file():
        log(f"Could not find file at {p}, found something else.", 'e', verbose)
        return False

    try:
        available_memory = psutil.virtual_memory().available
        file_size = os.path.getsize(file_path)
        if file_size > float(available_memory) * 0.9:
            recommended_gb = round(float(file_size) * 1.5 / (1024 * 1024 * 1024), 3)
            log(f"File size (almost) exceeds system memory. Recommended is at least "
                f"{recommended_gb} GB.", 'w', verbose)

    except Exception as e:
        log(f"Could not check memory requirements. Exception {e}", 'w', verbose)

    return True


if __name__ == '__main__':
    colorama_init()
    if len(sys.argv) > 1:
        # start_time = time.time()
        execute()
        # elapsed = round(time.time() - start_time, 3)
        # print("Finished in ", elapsed, " seconds.")
    else:
        print("No arguments given.")
        time.sleep(5)
