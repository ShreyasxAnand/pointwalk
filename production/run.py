"""Build a point cloud from converted Waymo data.

    python production/run.py                                  list the segments in data/
    python production/run.py <segment>                        frame 0, top laser
    python production/run.py <segment> --frame 5 --laser 1

data/<segment>/ must already hold step 1's output (production/waymo/convert.py).
This never downloads anything. The point cloud goes to
output/<segment>/frameFFFF_laserL_vehicle.bin, a point-cloud file (pointcloud.h),
in the vehicle frame.

Compiles production/adapters/waymo_to_points.cu into production/build/ first,
but only when a source file is newer than the program.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTPUT = os.path.join(HERE, "output")
BUILD = os.path.join(HERE, "build")
SOURCE = os.path.join(HERE, "adapters", "waymo_to_points.cu")
PROGRAM = os.path.join(BUILD, "waymo_to_points")

# nvcc on this machine comes with the PyTorch env (see CLAUDE.md). Its libraries
# are in lib/, not lib64/, hence the -L.
CUDA = "/opt/pytorch/cuda"
NVCC = shutil.which("nvcc") or os.path.join(CUDA, "bin", "nvcc")
FLAGS = ["-O2", "-arch=sm_75", "-fmad=false", "-L" + os.path.join(CUDA, "lib")]


def frames_of(segment):
    """{laser: sorted frame numbers} that have a range image in data/<segment>/range/."""
    found = {}
    for path in glob.glob(os.path.join(DATA, segment, "range", "frame*_laser*.bin")):
        m = re.match(r"frame(\d+)_laser(\d+)\.bin$", os.path.basename(path))
        if m:
            found.setdefault(int(m.group(2)), []).append(int(m.group(1)))
    return {laser: sorted(frames) for laser, frames in found.items()}


def list_segments():
    segments = sorted(d for d in os.listdir(DATA)
                      if os.path.exists(os.path.join(DATA, d, "calibration.bin")))
    if not segments:
        print(f"No converted segments in {DATA}. Run production/waymo/convert.py first.")
        return
    for segment in segments:
        frames = frames_of(segment).get(1, [])
        span = f"frames {frames[0]} to {frames[-1]} ({len(frames)})" if frames else "no frames"
        print(f"{segment}   laser 1: {span}")


def build():
    """Compile waymo_to_points if it is missing or older than any production source."""
    sources = [SOURCE] + glob.glob(os.path.join(HERE, "*", "*.h")) + glob.glob(os.path.join(HERE, "*", "*.cuh"))
    if os.path.exists(PROGRAM) and os.path.getmtime(PROGRAM) >= max(map(os.path.getmtime, sources)):
        return
    os.makedirs(BUILD, exist_ok=True)
    cmd = [NVCC, *FLAGS, SOURCE, "-o", PROGRAM]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Converted Waymo data -> point cloud")
    parser.add_argument("segment", nargs="?", help="a folder name in production/data/; leave out to list them")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--laser", type=int, default=1)
    args = parser.parse_args()

    if args.segment is None:
        list_segments()
        return 0

    folder = os.path.join(DATA, args.segment)
    if not os.path.exists(os.path.join(folder, "calibration.bin")):
        sys.exit(f"Not converted: {folder}. Run production/waymo/convert.py first.")
    if args.frame not in frames_of(args.segment).get(args.laser, []):
        sys.exit(f"No range image for frame {args.frame}, laser {args.laser} in {folder}/range/")

    build()
    os.makedirs(os.path.join(OUTPUT, args.segment), exist_ok=True)
    out = os.path.join(OUTPUT, args.segment, f"frame{args.frame:04d}_laser{args.laser}_vehicle.bin")
    cmd = [PROGRAM, folder, str(args.frame), str(args.laser), out]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
