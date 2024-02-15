import numpy as np
import pye57


class PointCloud:
    def __init__(self):
        self.points: np.ndarray = None
        self.intensities: np.ndarray = None
        self.colors: np.ndarray = None

    def read_e57(self, filename: str):
        e57 = pye57.E57(filename)
        raw_data = e57.read_scan_raw(0)

        x = raw_data["cartesianX"]
        y = raw_data["cartesianY"]
        z = raw_data["cartesianZ"]
        self.points = np.stack((x, y, z), axis=1)
        self.intensities = raw_data["intensity"]
        r = raw_data["colorRed"]
        g = raw_data["colorGreen"]
        b = raw_data["colorBlue"]
        self.colors = np.stack((r, g, b), axis=1)

    def write_e57(self, filename: str):
        e57 = pye57.E57(filename, mode="w")

        raw_data = dict()
        raw_data["cartesianX"] = self.points[:, 0]
        raw_data["cartesianY"] = self.points[:, 1]
        raw_data["cartesianZ"] = self.points[:, 2]

        raw_data["intensity"] = self.intensities

        raw_data["colorRed"] = self.colors[:, 0]
        raw_data["colorGreen"] = self.colors[:, 1]
        raw_data["colorBlue"] = self.colors[:, 2]

        e57.write_scan_raw(raw_data)