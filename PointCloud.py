import numpy as np
import pye57  # https://pypi.org/project/pye57/
import laspy  # https://pypi.org/project/laspy/


class PointCloud:
    def __init__(self):
        self.points: np.ndarray = None
        self.intensities: np.ndarray = None
        self.colors: np.ndarray = None

    # https://pypi.org/project/pye57/
    def read_e57(self, filename: str):
        e57 = pye57.E57(filename)
        raw_data = e57.read_scan_raw(0)

        x = raw_data["cartesianX"]
        y = raw_data["cartesianY"]
        z = raw_data["cartesianZ"]
        self.points = np.stack((x, y, z), axis=1)
        # TODO read some kind of scaling value so we can convert it correctly.

        if "intensity" in raw_data.keys():
            self.intensities = raw_data["intensity"]

        if "colorRed" in raw_data.keys() and "colorGreen" in raw_data.keys() and "colorBlue" in raw_data.keys():
            color_normalization_factor: float = 255.0  # TODO, normalize by value read from header or data.
            r = raw_data["colorRed"].astype(np.float64)
            g = raw_data["colorGreen"].astype(np.float64)
            b = raw_data["colorBlue"].astype(np.float64)
            self.colors = np.stack((r, g, b), axis=1) / color_normalization_factor

    # https://pypi.org/project/pye57/
    def write_e57(self, filename: str):
        e57 = pye57.E57(filename, mode="w")

        raw_data = dict()
        raw_data["cartesianX"] = self.points[:, 0]
        raw_data["cartesianY"] = self.points[:, 1]
        raw_data["cartesianZ"] = self.points[:, 2]

        if self.intensities is not None:
            raw_data["intensity"] = self.intensities

        if self.colors is not None:
            # e57 expects the color data to be ints between 0 and 255 (incl.)
            raw_data["colorRed"] = (self.colors[:, 0] * 255).astype(np.int32)
            raw_data["colorGreen"] = (self.colors[:, 1] * 255).astype(np.int32)
            raw_data["colorBlue"] = (self.colors[:, 2] * 255).astype(np.int32)

        e57.write_scan_raw(raw_data)

    # https://github.com/laspy/laspy/blob/740153c7b75abbea240d0b18a07f03038469f1fd/docs/complete_tutorial.rst#L76
    def read_las(self, filename: str):
        las = laspy.read(filename)

        has_rgb = 0
        rgb_normalization_factor = 1.0
        has_intensities = False
        for dimension in las.point_format.dimensions:
            n = dimension.name.lower()
            if n == "red" or n == "green" or n == "blue":
                has_rgb += 1
                rgb_normalization_factor = float(dimension.max)
            if n == "intensity" or n == "intensities":
                has_intensities = True

        point_count = int(las.header.point_count)

        # TODO read the scaling value so we can convert correctly
        self.points = las.xyz  # We can directly assign this, since the getter returns a new array

        if has_rgb:
            r = np.array(las.red)
            g = np.array(las.green)
            b = np.array(las.blue)
            self.colors = np.stack((r, g, b), axis=1).astype(np.float64)
            if rgb_normalization_factor != 1.0:  # Make sure to normalize the colors to a [0-1] range.
                self.colors /= rgb_normalization_factor

        if has_intensities:
            self.intensities = np.array(las.intensity)

    def write_las(self, filename: str):
        point_format = 3
        header = laspy.LasHeader(point_format=point_format)

        header.offsets = np.min(self.points, axis=0)
        header.scales = np.max(self.points - header.offsets, axis=0) / np.iinfo(np.int32).max
        las = laspy.LasData(header=header)

        # TODO properly write the data. We cannot handle fractional values well.
        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        int16_max = np.iinfo(np.uint16).max

        if self.intensities is not None:
            # Intensity is always unsigned 16bit with las, meaning the max value is 65535
            las.intensity = (self.intensities * int16_max).astype(np.uint16)
        else:
            las.intensity = np.zeros(shape=(len(self.points),), dtype=np.uint16)

        if self.colors is not None:
            # Like intensity, colors are unsigned 16bit, so we need to normalize to [0-65536]
            las.red = (self.colors[:, 0] * int16_max).astype(np.uint16)
            las.green = (self.colors[:, 1] * int16_max).astype(np.uint16)
            las.blue = (self.colors[:, 2] * int16_max).astype(np.uint16)

        las.write(filename)
