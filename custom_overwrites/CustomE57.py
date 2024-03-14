import uuid
from typing import Dict

import numpy as np
import pye57
from pye57 import libe57
from pye57.__version__ import __version__
from pye57.e57 import SUPPORTED_POINT_FIELDS
from tqdm import tqdm


class CustomE57(pye57.E57):
    def __init__(self, path, mode, verbose: int):
        super().__init__(path, mode)
        self.verbosity = verbose

    def write_scan_raw(self, data: Dict, *, name=None, rotation=None, translation=None, scan_header=None):
        for field in data.keys():
            if field not in SUPPORTED_POINT_FIELDS:
                raise ValueError("Unsupported point field: %s" % field)

        if rotation is None:
            rotation = getattr(scan_header, "rotation", np.array([1, 0, 0, 0]))

        if translation is None:
            translation = getattr(scan_header, "translation", np.array([0, 0, 0]))

        if name is None:
            name = getattr(scan_header, "name", "Scan %s" % len(self.data3d))

        temperature = getattr(scan_header, "temperature", 0)
        relativeHumidity = getattr(scan_header, "relativeHumidity", 0)
        atmosphericPressure = getattr(scan_header, "atmosphericPressure", 0)

        scan_node = libe57.StructureNode(self.image_file)
        scan_node.set("guid", libe57.StringNode(self.image_file, "{%s}" % uuid.uuid4()))
        scan_node.set("name", libe57.StringNode(self.image_file, name))
        scan_node.set("temperature", libe57.FloatNode(self.image_file, temperature))
        scan_node.set("relativeHumidity", libe57.FloatNode(self.image_file, relativeHumidity))
        scan_node.set("atmosphericPressure", libe57.FloatNode(self.image_file, atmosphericPressure))
        scan_node.set("description", libe57.StringNode(self.image_file, "pye57 v%s" % __version__))

        n_points = data["cartesianX"].shape[0]

        ibox = libe57.StructureNode(self.image_file)
        if "rowIndex" in data and "columnIndex" in data:
            min_row = np.min(data["rowIndex"])
            max_row = np.max(data["rowIndex"])
            min_col = np.min(data["columnIndex"])
            max_col = np.max(data["columnIndex"])
            ibox.set("rowMinimum", libe57.IntegerNode(self.image_file, min_row))
            ibox.set("rowMaximum", libe57.IntegerNode(self.image_file, max_row))
            ibox.set("columnMinimum", libe57.IntegerNode(self.image_file, min_col))
            ibox.set("columnMaximum", libe57.IntegerNode(self.image_file, max_col))
        else:
            ibox.set("rowMinimum", libe57.IntegerNode(self.image_file, 0))
            ibox.set("rowMaximum", libe57.IntegerNode(self.image_file, n_points - 1))
            ibox.set("columnMinimum", libe57.IntegerNode(self.image_file, 0))
            ibox.set("columnMaximum", libe57.IntegerNode(self.image_file, 0))
        ibox.set("returnMinimum", libe57.IntegerNode(self.image_file, 0))
        ibox.set("returnMaximum", libe57.IntegerNode(self.image_file, 0))
        scan_node.set("indexBounds", ibox)

        if "intensity" in data:
            int_min = getattr(scan_header, "intensityMinimum", np.min(data["intensity"]))
            int_max = getattr(scan_header, "intensityMaximum", np.max(data["intensity"]))
            intbox = libe57.StructureNode(self.image_file)
            intbox.set("intensityMinimum", libe57.FloatNode(self.image_file, int_min))
            intbox.set("intensityMaximum", libe57.FloatNode(self.image_file, int_max))
            scan_node.set("intensityLimits", intbox)

        color = all(c in data for c in ["colorRed", "colorGreen", "colorBlue"])
        if color:
            colorbox = libe57.StructureNode(self.image_file)
            colorbox.set("colorRedMinimum", libe57.IntegerNode(self.image_file, 0))
            colorbox.set("colorRedMaximum", libe57.IntegerNode(self.image_file, 255))
            colorbox.set("colorGreenMinimum", libe57.IntegerNode(self.image_file, 0))
            colorbox.set("colorGreenMaximum", libe57.IntegerNode(self.image_file, 255))
            colorbox.set("colorBlueMinimum", libe57.IntegerNode(self.image_file, 0))
            colorbox.set("colorBlueMaximum", libe57.IntegerNode(self.image_file, 255))
            scan_node.set("colorLimits", colorbox)

        bbox_node = libe57.StructureNode(self.image_file)
        x, y, z = data["cartesianX"], data["cartesianY"], data["cartesianZ"]
        valid = None
        if "cartesianInvalidState" in data:
            valid = ~data["cartesianInvalidState"].astype("?")
            x, y, z = x[valid], y[valid], z[valid]
        bb_min = np.array([x.min(), y.min(), z.min()])
        bb_max = np.array([x.max(), y.max(), z.max()])
        del valid, x, y, z

        if scan_header is not None:
            bb_min_scaled = np.array([scan_header.xMinimum, scan_header.yMinimum, scan_header.zMinimum])
            bb_max_scaled = np.array([scan_header.xMaximum, scan_header.yMaximum, scan_header.zMaximum])
        else:
            bb_min_scaled = self.to_global(bb_min.reshape(-1, 3), rotation, translation)[0]
            bb_max_scaled = self.to_global(bb_max.reshape(-1, 3), rotation, translation)[0]

        bbox_node.set("xMinimum", libe57.FloatNode(self.image_file, bb_min_scaled[0]))
        bbox_node.set("xMaximum", libe57.FloatNode(self.image_file, bb_max_scaled[0]))
        bbox_node.set("yMinimum", libe57.FloatNode(self.image_file, bb_min_scaled[1]))
        bbox_node.set("yMaximum", libe57.FloatNode(self.image_file, bb_max_scaled[1]))
        bbox_node.set("zMinimum", libe57.FloatNode(self.image_file, bb_min_scaled[2]))
        bbox_node.set("zMaximum", libe57.FloatNode(self.image_file, bb_max_scaled[2]))
        scan_node.set("cartesianBounds", bbox_node)

        if rotation is not None and translation is not None:
            pose_node = libe57.StructureNode(self.image_file)
            scan_node.set("pose", pose_node)
            rotation_node = libe57.StructureNode(self.image_file)
            rotation_node.set("w", libe57.FloatNode(self.image_file, rotation[0]))
            rotation_node.set("x", libe57.FloatNode(self.image_file, rotation[1]))
            rotation_node.set("y", libe57.FloatNode(self.image_file, rotation[2]))
            rotation_node.set("z", libe57.FloatNode(self.image_file, rotation[3]))
            pose_node.set("rotation", rotation_node)
            translation_node = libe57.StructureNode(self.image_file)
            translation_node.set("x", libe57.FloatNode(self.image_file, translation[0]))
            translation_node.set("y", libe57.FloatNode(self.image_file, translation[1]))
            translation_node.set("z", libe57.FloatNode(self.image_file, translation[2]))
            pose_node.set("translation", translation_node)

        start_datetime = getattr(scan_header, "acquisitionStart_dateTimeValue", 0)
        start_atomic = getattr(scan_header, "acquisitionStart_isAtomicClockReferenced", False)
        end_datetime = getattr(scan_header, "acquisitionEnd_dateTimeValue", 0)
        end_atomic = getattr(scan_header, "acquisitionEnd_isAtomicClockReferenced", False)
        acquisition_start = libe57.StructureNode(self.image_file)
        scan_node.set("acquisitionStart", acquisition_start)
        acquisition_start.set("dateTimeValue", libe57.FloatNode(self.image_file, start_datetime))
        acquisition_start.set("isAtomicClockReferenced", libe57.IntegerNode(self.image_file, start_atomic))
        acquisition_end = libe57.StructureNode(self.image_file)
        scan_node.set("acquisitionEnd", acquisition_end)
        acquisition_end.set("dateTimeValue", libe57.FloatNode(self.image_file, end_datetime))
        acquisition_end.set("isAtomicClockReferenced", libe57.IntegerNode(self.image_file, end_atomic))

        # todo: pointGroupingSchemes

        points_prototype = libe57.StructureNode(self.image_file)

        is_scaled = False
        precision = libe57.E57_DOUBLE if is_scaled else libe57.E57_SINGLE

        center = (bb_max + bb_min) / 2

        chunk_size = 1_000_000

        x_node = libe57.FloatNode(self.image_file, center[0], precision, bb_min[0], bb_max[0])
        y_node = libe57.FloatNode(self.image_file, center[1], precision, bb_min[1], bb_max[1])
        z_node = libe57.FloatNode(self.image_file, center[2], precision, bb_min[2], bb_max[2])
        points_prototype.set("cartesianX", x_node)
        points_prototype.set("cartesianY", y_node)
        points_prototype.set("cartesianZ", z_node)

        field_names = ["cartesianX", "cartesianY", "cartesianZ"]

        if "intensity" in data:
            intensity_min = np.min(data["intensity"])
            intensity_max = np.max(data["intensity"])
            intensity_node = libe57.FloatNode(self.image_file, intensity_min, precision, intensity_min, intensity_max)
            points_prototype.set("intensity", intensity_node)
            field_names.append("intensity")

        if all(color in data for color in ["colorRed", "colorGreen", "colorBlue"]):
            points_prototype.set("colorRed", libe57.IntegerNode(self.image_file, 0, 0, 255))
            points_prototype.set("colorGreen", libe57.IntegerNode(self.image_file, 0, 0, 255))
            points_prototype.set("colorBlue", libe57.IntegerNode(self.image_file, 0, 0, 255))
            field_names.append("colorRed")
            field_names.append("colorGreen")
            field_names.append("colorBlue")

        if "rowIndex" in data and "columnIndex" in data:
            min_row = np.min(data["rowIndex"])
            max_row = np.max(data["rowIndex"])
            min_col = np.min(data["columnIndex"])
            max_col = np.max(data["columnIndex"])
            points_prototype.set("rowIndex", libe57.IntegerNode(self.image_file, 0, min_row, max_row))
            field_names.append("rowIndex")
            points_prototype.set("columnIndex", libe57.IntegerNode(self.image_file, 0, min_col, max_col))
            field_names.append("columnIndex")

        if "cartesianInvalidState" in data:
            min_state = np.min(data["cartesianInvalidState"])
            max_state = np.max(data["cartesianInvalidState"])
            points_prototype.set("cartesianInvalidState", libe57.IntegerNode(self.image_file, 0, min_state, max_state))
            field_names.append("cartesianInvalidState")

        # other fields
        # // "sphericalRange"
        # // "sphericalAzimuth"
        # // "sphericalElevation"
        # // "timeStamp"
        # // "sphericalInvalidState"
        # // "isColorInvalid"
        # // "isIntensityInvalid"
        # // "isTimeStampInvalid"

        arrays, buffers = self.make_buffers(field_names, chunk_size)

        codecs = libe57.VectorNode(self.image_file, True)
        points = libe57.CompressedVectorNode(self.image_file, points_prototype, codecs)
        scan_node.set("points", points)

        self.data3d.append(scan_node)

        writer = points.writer(buffers)

        progress_bar = None
        if self.verbosity == 2:
            progress_bar = tqdm(desc="Writing progress", unit='%', total=100)

        current_index = 0
        progress = 0
        while current_index != n_points:
            current_chunk = min(n_points - current_index, chunk_size)

            for type_ in SUPPORTED_POINT_FIELDS:
                if type_ in arrays:
                    arrays[type_][:current_chunk] = data[type_][current_index:current_index + current_chunk]

            writer.write(current_chunk)

            new_progress = int(current_index / n_points * 100)
            progress_increase = int(new_progress - progress)
            progress = new_progress
            if progress_bar:
                progress_bar.update(progress_increase)
            elif self.verbosity == 3 and progress_increase > 0:
                print(f"w{progress}")

            current_index += current_chunk

        if self.verbosity == 3:
            print("w100")
        elif progress_bar:
            progress_bar.update(float(100.0 - progress))

        writer.close()
