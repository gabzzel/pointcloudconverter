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

        self.intensities = raw_data["intensity"]

        # TODO, normalize by value read from header or data.
        color_normalization_factor: float = 255.0
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

        raw_data["intensity"] = self.intensities

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
        self.points = np.array(las.xyz)
        self.intensities = np.zeros(shape=(point_count, ))
        self.colors = np.zeros(shape=(point_count, 3))

        if has_rgb:
            r = np.array(las.red)
            g = np.array(las.green)
            b = np.array(las.blue)
            self.colors = np.stack((r, g, b), axis=1).astype(np.float64)
            if rgb_normalization_factor != 1.0: # Make sure to normalize the colors to a [0-1] range.
                self.colors /= rgb_normalization_factor

        if has_intensities:
            self.intensities = np.array(las.intensity)