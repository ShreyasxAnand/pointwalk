"""Convert one Waymo Open Dataset v2 segment from Parquet to plain binary files.

    python convert.py --input <folder> --segment <id> --out <dir>

<folder> holds lidar/, lidar_calibration/ and vehicle_pose/. It can be a local
folder, gs://... or s3://...

What gets written (the byte layout is in DATA_FORMAT.md):

    <out>/calibration.bin                    every laser: extrinsic, beam inclinations
    <out>/poses.bin                          every frame: timestamp, vehicle pose
    <out>/range/frame0000_laser1.bin         one range image: first return, all channels
    <out>/range/frame0000_laser2.bin         ...

No number is changed on the way: same order, same axes, same units as Waymo
stores them. This file does no geometry.
"""

import argparse
import os
import struct

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import wod_tables as wod


def write_calibration(fs, root, segment, out_dir):
    """Write calibration.bin. Returns the laser names, smallest first."""
    path = wod.require_table(fs, root, "lidar_calibration", segment)
    columns = [wod.KEY_LASER, wod.CALIB_EXTRINSIC, wod.CALIB_BEAM_MIN,
               wod.CALIB_BEAM_MAX, wod.CALIB_BEAM_VALUES]
    table = pd.read_parquet(path, columns=columns, filesystem=fs)
    table = table.sort_values(wod.KEY_LASER)

    with open(os.path.join(out_dir, "calibration.bin"), "wb") as f:
        f.write(struct.pack("<8si", b"PWCALIB1", len(table)))
        for _, row in table.iterrows():
            extrinsic = np.asarray(row[wod.CALIB_EXTRINSIC], dtype="<f8")
            if extrinsic.size != 16:
                raise SystemExit(f"laser {row[wod.KEY_LASER]}: extrinsic has {extrinsic.size} numbers, expected 16")
            # Only some lasers carry a table of beam angles. The others have null
            # here and give just a smallest and a largest angle.
            values = row[wod.CALIB_BEAM_VALUES]
            beams = np.zeros(0, dtype="<f8") if values is None else np.asarray(values, dtype="<f8")
            f.write(struct.pack("<ii", int(row[wod.KEY_LASER]), beams.size))
            f.write(extrinsic.tobytes())
            f.write(struct.pack("<dd", row[wod.CALIB_BEAM_MIN], row[wod.CALIB_BEAM_MAX]))
            f.write(beams.tobytes())
    return [int(x) for x in table[wod.KEY_LASER]]


def write_poses(fs, root, segment, out_dir):
    """Write poses.bin. Returns {timestamp: frame number}, frames in time order."""
    path = wod.require_table(fs, root, "vehicle_pose", segment)
    table = pd.read_parquet(path, columns=[wod.KEY_TIMESTAMP, wod.POSE_TRANSFORM], filesystem=fs)
    table = table.sort_values(wod.KEY_TIMESTAMP)
    if table[wod.KEY_TIMESTAMP].duplicated().any():
        raise SystemExit("vehicle_pose has two rows with the same timestamp")

    frame_of = {}
    with open(os.path.join(out_dir, "poses.bin"), "wb") as f:
        f.write(struct.pack("<8si", b"PWPOSES1", len(table)))
        for frame, (_, row) in enumerate(table.iterrows()):
            timestamp = int(row[wod.KEY_TIMESTAMP])
            transform = np.asarray(row[wod.POSE_TRANSFORM], dtype="<f8")
            if transform.size != 16:
                raise SystemExit(f"frame {frame}: pose has {transform.size} numbers, expected 16")
            f.write(struct.pack("<q", timestamp))
            f.write(transform.tobytes())
            frame_of[timestamp] = frame
    return frame_of


def write_range_images(fs, root, segment, out_dir, frame_of):
    """Write one file per (frame, laser). Returns {laser: (H, W, C)} and the byte total."""
    path = wod.require_table(fs, root, "lidar", segment)
    range_dir = os.path.join(out_dir, "range")
    os.makedirs(range_dir, exist_ok=True)

    columns = [wod.KEY_TIMESTAMP, wod.KEY_LASER, wod.LIDAR_VALUES, wod.LIDAR_SHAPE]
    shapes = {}
    written = set()
    total_bytes = 0
    with fs.open(path, "rb") as source:
        parquet = pq.ParquetFile(source)
        # batch_size=1: one row, which is one laser of one frame, is in memory
        # at a time. Only the four columns above are read from the file.
        for batch in parquet.iter_batches(batch_size=1, columns=columns):
            timestamp = batch.column(wod.KEY_TIMESTAMP)[0].as_py()
            laser = batch.column(wod.KEY_LASER)[0].as_py()
            shape = batch.column(wod.LIDAR_SHAPE)[0].as_py()
            values = batch.column(wod.LIDAR_VALUES).flatten().to_numpy(zero_copy_only=False)

            if timestamp not in frame_of:
                raise SystemExit(f"lidar row at timestamp {timestamp} has no row in vehicle_pose")
            frame = frame_of[timestamp]
            H, W, C = shape
            if values.size != H * W * C:
                raise SystemExit(f"frame {frame} laser {laser}: {values.size} values, but shape {shape} needs {H * W * C}")
            if (frame, laser) in written:
                raise SystemExit(f"frame {frame} laser {laser} appears twice in the lidar table")
            written.add((frame, laser))
            if shapes.setdefault(laser, (H, W, C)) != (H, W, C):
                raise SystemExit(f"laser {laser} changes shape: {shapes[laser]} then {(H, W, C)}")

            values = np.ascontiguousarray(values, dtype="<f4")
            name = f"frame{frame:04d}_laser{laser}.bin"
            with open(os.path.join(range_dir, name), "wb") as f:
                f.write(struct.pack("<8sqiiiii", b"PWRANGE1", timestamp, frame, laser, H, W, C))
                f.write(values.tobytes())
            total_bytes += 36 + values.nbytes

            if laser == min(shapes) and frame % 20 == 0:
                print(f"  frame {frame} of {len(frame_of)}", flush=True)
    return shapes, written, total_bytes


def main():
    parser = argparse.ArgumentParser(description="Waymo v2 segment (Parquet) -> plain binary files. Layout: DATA_FORMAT.md")
    parser.add_argument("--input", required=True, help="folder holding lidar/, lidar_calibration/, vehicle_pose/ (local, gs:// or s3://)")
    parser.add_argument("--segment", required=True, help="segment id: a file name in those folders, without .parquet")
    parser.add_argument("--out", required=True, help="local folder to write into (created if missing)")
    args = parser.parse_args()

    fs, root = wod.open_dataset(args.input)
    # Stop before creating anything if one of the three tables is missing.
    for tag in ("lidar_calibration", "vehicle_pose", "lidar"):
        wod.require_table(fs, root, tag, args.segment)
    os.makedirs(args.out, exist_ok=True)

    lasers = write_calibration(fs, root, args.segment, args.out)
    print(f"calibration.bin: {len(lasers)} lasers {lasers}")

    frame_of = write_poses(fs, root, args.segment, args.out)
    print(f"poses.bin: {len(frame_of)} frames")

    shapes, written, total_bytes = write_range_images(fs, root, args.segment, args.out, frame_of)
    for laser in sorted(shapes):
        H, W, C = shapes[laser]
        count = sum(1 for (_, l) in written if l == laser)
        print(f"range/: laser {laser}: {count} files, H {H}, W {W}, C {C}")
    print(f"range/: {len(written)} files, {total_bytes / 1e6:.1f} MB")

    # Say so if the three tables disagree about which frames and lasers exist.
    missing = [(frame, laser) for frame in frame_of.values() for laser in lasers
               if (frame, laser) not in written]
    if missing:
        print(f"WARNING: {len(missing)} (frame, laser) pairs have no range image, first: {missing[:5]}")
    extra = sorted(set(shapes) - set(lasers))
    if extra:
        print(f"WARNING: lasers {extra} have range images but no calibration")


if __name__ == "__main__":
    main()
