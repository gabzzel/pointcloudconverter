import pathlib
from typing import BinaryIO, Optional, Sequence, Union

import laspy
from laspy import LazBackend

from custom_overwrites.LasWriteOverwrite import CustomLasWriter


class CustomLasData(laspy.LasData):
    def write(self, destination, do_compress=None, laz_backend=None, verbose: int = 0):
        if do_compress:
            super().write(destination, do_compress, laz_backend)
        else:
            if isinstance(destination, (str, pathlib.Path)):
                with open(destination, mode="wb+") as out:
                    self._write_to(out, do_compress=do_compress, laz_backend=laz_backend, verbose=verbose)
            else:
                self._write_to(
                    destination, do_compress=do_compress, laz_backend=laz_backend
                )

    def _write_to(
        self,
        out_stream: BinaryIO,
        do_compress: Optional[bool] = None,
        laz_backend: Optional[Union[LazBackend, Sequence[LazBackend]]] = None,
        verbose: int = 0,
    ) -> None:
        with CustomLasWriter(
            out_stream,
            self.header,
            verbose
        ) as writer:
            writer.write_points(self.points)
            if self.header.version.minor >= 4 and self.evlrs is not None:
                writer.write_evlrs(self.evlrs)
