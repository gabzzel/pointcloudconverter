from typing import BinaryIO

import pypcd4
from tqdm import tqdm


class CustomPcdPointCloud(pypcd4.PointCloud):
    def __init__(self, metadata, pc_data, verbosity):
        super().__init__(metadata, pc_data)
        self.verbosity = verbosity

    def _save_as_binary(self, fp: BinaryIO) -> None:
        # fp.write(self.pc_data.tobytes())

        if self.verbosity != 2 and self.verbosity != 3:
            super()._save_as_binary(fp)
            return

        progress_bar = None

        if self.verbosity == 2:
            progress_bar = tqdm(desc="Writing progress", total=100, unit='%')

        data = self.pc_data.tobytes()
        chunk_size = 1024
        total_size = len(data)
        bytes_written = 0
        progress = 0

        # Write data to the BytesIO object in chunks
        for chunk_start in range(0, total_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_size)
            chunk_data = data[chunk_start:chunk_end]
            fp.write(chunk_data)

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
