import math

import plyfile
import tqdm


class CustomPlyElement(plyfile.PlyElement):
    def __init__(self, name, properties, count, comments, verbose):
        super().__init__(name, properties, count, comments)
        self.verbosity = verbose

    def _write(self, stream, text, byte_order):
        if text:
            super()._write(stream, text, byte_order)
        else:
            if self._have_list:
                # There are list properties, so serialization is
                # slightly complicated.
                self._write_bin(stream, byte_order)
            else:
                # no list properties, so serialization is
                # straightforward.
                self.write_with_progress(stream, self.data.astype(self.dtype(byte_order), copy=False).data)
                # stream.write(self.data.astype(self.dtype(byte_order), copy=False).data)

    def write_with_progress(self, stream, data):
        if self.verbosity != 2 and self.verbosity != 3:
            stream.write(data)
            return

        progress_bar = None

        if self.verbosity == 2:
            progress_bar = tqdm.tqdm(desc="Writing progress", total=100, unit='%')

        chunk_size = 1024
        total_size = len(data)
        bytes_written = 0
        progress = 0

        # Write data to the BytesIO object in chunks
        for chunk_start in range(0, total_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_size)
            chunk_data = data[chunk_start:chunk_end]
            stream.write(chunk_data)

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

    def _write_bin(self, stream, byte_order):
        if self.verbosity != 2 and self.verbosity != 3:
            super()._write_bin(stream, byte_order)
            return

        progress_bar = None

        if self.verbosity == 2:
            progress_bar = tqdm.tqdm(desc="Writing progress", total=100, unit='%')

        for i in range(len(self.data)):
            rec = self.data[i]
            for prop in self.properties:
                prop._write_bin(rec[prop.name], stream, byte_order)

            if progress_bar:
                progress_bar.update()
            else:
                progress = math.ceil(i / len(self.data))
                print(f"w{progress}")
