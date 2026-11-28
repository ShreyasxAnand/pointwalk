"""Names and paths of the Waymo Open Dataset v2 tables. Shared by convert.py
and inspect_segment.py so both spell the column names the same way.

The dataset is a folder of table folders, one Parquet file per segment:

    <input>/lidar/<segment>.parquet
    <input>/lidar_calibration/<segment>.parquet
    <input>/vehicle_pose/<segment>.parquet

<input> can be a local folder, gs://bucket/folder or s3://bucket/folder.
"""

import fsspec

# Key columns: the same names in every table.
KEY_SEGMENT = "key.segment_context_name"
KEY_TIMESTAMP = "key.frame_timestamp_micros"
KEY_LASER = "key.laser_name"

# lidar: one row per (frame, laser).
LIDAR_VALUES = "[LiDARComponent].range_image_return1.values"   # flat list of float32
LIDAR_SHAPE = "[LiDARComponent].range_image_return1.shape"     # [H, W, C]

# lidar_calibration: one row per laser.
CALIB_EXTRINSIC = "[LiDARCalibrationComponent].extrinsic.transform"        # 16 doubles
CALIB_BEAM_MIN = "[LiDARCalibrationComponent].beam_inclination.min"
CALIB_BEAM_MAX = "[LiDARCalibrationComponent].beam_inclination.max"
CALIB_BEAM_VALUES = "[LiDARCalibrationComponent].beam_inclination.values"  # list of doubles, or null

# vehicle_pose: one row per frame.
POSE_TRANSFORM = "[VehiclePoseComponent].world_from_vehicle.transform"     # 16 doubles


def open_dataset(input_path):
    """Turn the --input text into (filesystem, folder path).

    fsspec picks the filesystem from the prefix: none = local disk, gs:// =
    Google Cloud Storage (needs gcsfs), s3:// = Amazon S3 (needs s3fs). Login is
    whatever that library finds on the machine; nothing here is specific to
    Colab, AWS or a laptop.
    """
    fs, root = fsspec.core.url_to_fs(input_path)
    return fs, root.rstrip("/")


def table_path(root, tag, segment):
    """Path of one table's file for one segment."""
    return f"{root}/{tag}/{segment}.parquet"


def require_table(fs, root, tag, segment):
    """Return the table's path, or stop with a message that says what was tried."""
    path = table_path(root, tag, segment)
    if not fs.exists(path):
        raise SystemExit(
            f"Not found: {path}\n"
            f"--input must be the folder that holds lidar/, lidar_calibration/ and "
            f"vehicle_pose/, and --segment a file name in them without .parquet."
        )
    return path
