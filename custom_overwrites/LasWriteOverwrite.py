from typing import BinaryIO, overload

from laspy import LasHeader, PackedPointRecord
from laspy.laswriter import UncompressedPointWriter, LasWriter
import tqdm


class CustomLasWriter(LasWriter):
    def __init__(self, dest: BinaryIO, header: LasHeader, verbosity: int):
        super().__init__(dest, header)
        self.point_writer = CustomUncompressedPointWriter(self.dest, header, verbosity)


class CustomUncompressedPointWriter(UncompressedPointWriter):
    def __init__(self, dest: BinaryIO, header: LasHeader, verbosity: int):
        super().__init__(dest)
        self.verbosity = verbosity

    def write_points(self, points: PackedPointRecord) -> None:
        if self.verbosity != 2 and self.verbosity != 3:
            super().write_points(points)
            return

        progress_bar = None

        if self.verbosity == 2:
            progress_bar = tqdm.tqdm(desc="Writing las/laz progress", total=100, unit='%')

        memory_view = points.memoryview()
        chunk_size = 1024
        total_size = len(memory_view)
        bytes_written = 0
        progress = 0

        # Write data to the BytesIO object in chunks
        for chunk_start in range(0, total_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_size)
            chunk_data = memory_view[chunk_start:chunk_end]
            self.dest.write(chunk_data)

            # Update progress
            bytes_written += len(chunk_data)
            new_progress = (bytes_written / total_size) * 100

            # Print progress if it has changed significantly
            if int(new_progress) != progress:
                progress = int(new_progress)
                if progress_bar:
                    progress_bar.update()
                else:
                    print(f"w{progress}")
