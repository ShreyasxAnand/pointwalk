"""Print what is really in one Waymo v2 segment, to check DATA_FORMAT.md against.

    python inspect_segment.py --input <folder> --segment <id>
    python inspect_segment.py --input <folder>              (lists some segment ids)

<folder> holds lidar/, lidar_calibration/ and vehicle_pose/. It can be a local
folder, gs://... or s3://...

It reads the small tables whole, and from the big lidar table only the keys,
the shapes and the first frame's values.

(Named inspect_segment.py, not inspect.py: a file called inspect.py would hide
Python's own "inspect" module from pandas.)
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import wod_tables as wod

NEEDED = ["lidar_calibration", "vehicle_pose", "lidar"]
LATER = ["lidar_pose", "lidar_box", "camera_image", "camera_calibration"]


def print_columns(fs, root, tag, segment):
    """Print a table's size, row counts and columns. Reads only the file's footer."""
    path = wod.table_path(root, tag, segment)
    if not fs.exists(path):
        print(f"\n== {tag}: NOT FOUND at {path}")
        return False
    size = fs.info(path)["size"]
    with fs.open(path, "rb") as f:
        parquet = pq.ParquetFile(f)
        meta = parquet.metadata
        groups = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        print(f"\n== {tag}: {size / 1e6:.1f} MB, {meta.num_rows} rows, row groups {groups}")
        for name, kind in zip(parquet.schema_arrow.names, parquet.schema_arrow.types):
            print(f"   {str(kind):36s} {name}")
    return True


def print_matrix(numbers):
    """Print 16 numbers as 4 rows of 4, in the order they are stored."""
    for row in np.asarray(numbers, dtype=np.float64).reshape(4, 4):
        print("     " + " ".join(f"{x:14.6f}" for x in row))


def inspect_calibration(fs, root, segment):
    table = pd.read_parquet(wod.table_path(root, "lidar_calibration", segment), filesystem=fs)
    print(f"   lasers in file order: {[int(x) for x in table[wod.KEY_LASER]]}")
    for _, row in table.sort_values(wod.KEY_LASER).iterrows():
        print(f"   laser {int(row[wod.KEY_LASER])}")
        print("     extrinsic.transform, 16 numbers in stored order, 4 per line:")
        print_matrix(row[wod.CALIB_EXTRINSIC])
        print(f"     beam_inclination.min {row[wod.CALIB_BEAM_MIN]!r}  max {row[wod.CALIB_BEAM_MAX]!r}")
        values = row[wod.CALIB_BEAM_VALUES]
        if values is None:
            print("     beam_inclination.values: null (no table)")
        else:
            values = np.asarray(values)
            steps = np.diff(values)
            order = "rising" if (steps > 0).all() else "falling" if (steps < 0).all() else "mixed"
            print(f"     beam_inclination.values: {values.size} numbers, {values.dtype}, {order}")
            print(f"       first 3: {values[:3].tolist()}")
            print(f"       last 3:  {values[-3:].tolist()}")


def inspect_poses(fs, root, segment):
    table = pd.read_parquet(wod.table_path(root, "vehicle_pose", segment), filesystem=fs)
    stamps = table[wod.KEY_TIMESTAMP].to_numpy()
    print(f"   frames: {len(table)}, in time order: {bool((np.diff(stamps) > 0).all())}")
    if len(stamps) > 1:
        steps = np.diff(np.sort(stamps))
        print(f"   time between frames, microseconds: smallest {steps.min()}, largest {steps.max()}")
    first = table.sort_values(wod.KEY_TIMESTAMP).iloc[0]
    print(f"   first frame, timestamp {int(first[wod.KEY_TIMESTAMP])}")
    print("     world_from_vehicle.transform, 16 numbers in stored order, 4 per line:")
    print_matrix(first[wod.POSE_TRANSFORM])
    return set(int(x) for x in stamps)


def inspect_lidar(fs, root, segment, pose_stamps):
    with fs.open(wod.table_path(root, "lidar", segment), "rb") as f:
        parquet = pq.ParquetFile(f)

        # Keys and shapes of every row: small columns, cheap to read.
        keys = parquet.read(columns=[wod.KEY_TIMESTAMP, wod.KEY_LASER, wod.LIDAR_SHAPE]).to_pandas()
        stamps = set(int(x) for x in keys[wod.KEY_TIMESTAMP])
        print(f"   rows: {len(keys)}, frames: {len(stamps)}, lasers: {sorted(int(x) for x in keys[wod.KEY_LASER].unique())}")
        print(f"   same timestamps as vehicle_pose: {stamps == pose_stamps}")
        for laser, group in keys.groupby(wod.KEY_LASER):
            shapes = sorted(set(tuple(int(x) for x in s) for s in group[wod.LIDAR_SHAPE]))
            print(f"   laser {int(laser)}: {len(group)} rows, shape [H, W, C] = {shapes}")

        # Values of the first frame only, one laser at a time.
        first_stamp = min(stamps)
        print(f"   first frame (timestamp {first_stamp}), first return:")
        columns = [wod.KEY_TIMESTAMP, wod.KEY_LASER, wod.LIDAR_VALUES, wod.LIDAR_SHAPE]
        seen = 0
        for batch in parquet.iter_batches(batch_size=1, columns=columns):
            if batch.column(wod.KEY_TIMESTAMP)[0].as_py() != first_stamp:
                continue
            laser = batch.column(wod.KEY_LASER)[0].as_py()
            H, W, C = batch.column(wod.LIDAR_SHAPE)[0].as_py()
            values = batch.column(wod.LIDAR_VALUES).flatten().to_numpy(zero_copy_only=False)
            fits = "matches H*W*C" if values.size == H * W * C else f"DOES NOT match H*W*C = {H * W * C}"
            print(f"     laser {laser}: {values.size} values, {values.dtype}, {fits}")
            if values.size == H * W * C:
                image = values.reshape(H, W, C)
                # The same two lines io/waymo/read_frame.cpp prints, to compare by eye.
                shown = min(W, 8)
                first = " ".join(f"{x:.4f}" for x in image[0, :shown, 0])
                print(f"       first {shown} ranges (channel 0, row 0, columns 0 to {shown - 1}): {first}")
                rows, cols = np.nonzero(image[:, :, 0] > 0)   # in stored order: row after row
                hits = " ".join(f"{r},{c}={image[r, c, 0]:.4f}" for r, c in zip(rows[:8], cols[:8]))
                print(f"       first ranges with a return, as row,col=range: {hits}")
                for c in range(C):
                    channel = image[:, :, c]
                    print(f"       channel {c}: smallest {channel.min():.4f}, largest {channel.max():.4f}, "
                          f"boxes > 0: {int((channel > 0).sum())}, boxes == -1: {int((channel == -1).sum())}")
            seen += 1
            if seen == keys[wod.KEY_LASER].nunique():
                break


def list_segments(fs, root):
    paths = sorted(fs.glob(f"{root}/lidar_calibration/*.parquet"))
    print(f"{len(paths)} segments in {root}/lidar_calibration. First 10:")
    for path in paths[:10]:
        print("  " + path.rsplit("/", 1)[-1][:-len(".parquet")])


def main():
    parser = argparse.ArgumentParser(description="Print the real columns and shapes of one Waymo v2 segment")
    parser.add_argument("--input", required=True, help="folder holding lidar/, lidar_calibration/, vehicle_pose/ (local, gs:// or s3://)")
    parser.add_argument("--segment", help="segment id; leave out to list some ids")
    args = parser.parse_args()

    fs, root = wod.open_dataset(args.input)
    if not args.segment:
        list_segments(fs, root)
        return

    print(f"segment {args.segment}")
    print(f"input   {args.input}")

    pose_stamps = set()
    if print_columns(fs, root, "lidar_calibration", args.segment):
        inspect_calibration(fs, root, args.segment)
    if print_columns(fs, root, "vehicle_pose", args.segment):
        pose_stamps = inspect_poses(fs, root, args.segment)
    if print_columns(fs, root, "lidar", args.segment):
        inspect_lidar(fs, root, args.segment, pose_stamps)

    print("\n---- tables for later: columns only ----")
    for tag in LATER:
        print_columns(fs, root, tag, args.segment)


if __name__ == "__main__":
    main()
