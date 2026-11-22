#!/usr/bin/env python3
"""Fetch real aerial lidar for a place and save it as a scene file for the viewer.

Source: the USGS 3DEP public lidar bucket on AWS (usgs-lidar-public). Each survey
there is stored as a tree of small files: the top of the tree is a thin sample of
the whole survey, and every level down adds more dots. So --depth sets the detail:
one level deeper is roughly 4 times the dots.

Example (Stanford Main Quad, 1 km square):
    python3 fetch_place.py --survey CA_SantaClaraCounty_2020 \
        --lat 37.4275 --lon -122.1700 --size 1000 --depth 11 --out scene.bin

Needs:  pip install "laspy[lazrs]" numpy
"""
import argparse
import io
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np

BUCKET = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
EARTH_RADIUS = 6378137.0     # metres, the one the web-map grid uses


def get(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as reply:
                return reply.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def to_web_map(lon, lat):
    """Longitude/latitude in degrees -> the survey's x, y (web-map metres)."""
    x = EARTH_RADIUS * math.radians(lon)
    y = EARTH_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def find_files(root, info, box, max_depth):
    """Walk the tree and list the files whose cube touches our box, down to max_depth."""
    x0, y0, z0, x1, y1, _ = info["bounds"]
    side = x1 - x0
    wanted = []

    def touches(depth, x, y):
        cell = side / (1 << depth)
        return not (x0 + (x + 1) * cell < box[0] or x0 + x * cell > box[2] or
                    y0 + (y + 1) * cell < box[1] or y0 + y * cell > box[3])

    def visit(key, table):
        depth, x, y, z = key
        name = "%d-%d-%d-%d" % key
        count = table.get(name)
        if count is None or not touches(depth, x, y):
            return
        if count == -1:                       # this branch's list lives in its own file
            table = json.loads(get(f"{root}/ept-hierarchy/{name}.json"))
            count = table.get(name, 0)
        if count > 0:
            wanted.append((name, count))
        if depth < max_depth:
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        visit((depth + 1, 2 * x + a, 2 * y + b, 2 * z + c), table)

    visit((0, 0, 0, 0), json.loads(get(f"{root}/ept-hierarchy/0-0-0-0.json")))
    return wanted


def read_file(root, name, box, centre, stretch):
    """Download one file, keep the dots inside our box, and return them in small local numbers.

    Returns east, north, up (metres, float32; east and north measured from the centre),
    brightness, label, echo count.
    """
    las = laspy.read(io.BytesIO(get(f"{root}/ept-data/{name}.laz")))
    x, y = np.asarray(las.x), np.asarray(las.y)
    keep = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    return (((x[keep] - centre[0]) / stretch).astype(np.float32),
            ((y[keep] - centre[1]) / stretch).astype(np.float32),
            np.asarray(las.z)[keep].astype(np.float32),
            np.asarray(las.intensity)[keep].astype(np.float32),
            np.asarray(las.classification)[keep].astype(np.uint8),
            np.asarray(las.number_of_returns)[keep].astype(np.uint8))


def colour(intensity, kind, returns):
    """No camera colour in this survey, so paint by what the laser tells us.

    brightness: how strongly the surface reflected the laser (roads dark, grass bright)
    tint:       the survey labels each dot (ground, plants, building, water, ...);
                each label gets its own colour
    """
    low, high = np.percentile(intensity, [2, 98])
    bright = np.clip((intensity - low) / max(high - low, 1.0), 0.0, 1.0)
    bright = 0.35 + 0.65 * bright ** 0.8

    tint = np.empty((len(kind), 3), dtype=np.float32)
    tint[:] = (200, 200, 205)                               # unlabelled: cars, walls, poles, wires
    tint[(kind == 1) & (returns > 1)] = (120, 165, 100)     # unlabelled with several echoes: likely leaves
    tint[kind == 2] = (196, 178, 140)                       # ground
    tint[kind == 3] = (130, 175, 90)                        # low plants
    tint[kind == 4] = (95, 160, 80)                         # bushes
    tint[kind == 5] = (70, 140, 70)                         # trees
    tint[kind == 6] = (238, 214, 190)                       # buildings
    tint[kind == 9] = (70, 120, 200)                        # water
    tint[kind == 17] = (170, 170, 175)                      # bridges
    return np.clip(tint * bright[:, None], 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Real lidar -> scene.bin")
    parser.add_argument("--survey", default="CA_SantaClaraCounty_2020")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--size", type=float, default=1000.0, help="side of the square, in metres")
    parser.add_argument("--depth", type=int, default=11, help="tree depth; one more is about 4x the dots")
    parser.add_argument("--out", default="scene.bin")
    parser.add_argument("--view-from", type=float, default=33.0,
                        help="compass direction the opening camera stands in, seen from the centre (degrees)")
    parser.add_argument("--view-distance", type=float, default=450.0, help="metres from the centre")
    parser.add_argument("--view-height", type=float, default=220.0, help="metres above the centre's ground")
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()

    root = f"{BUCKET}/{args.survey}"
    info = json.loads(get(f"{root}/ept.json"))
    if str(info.get("srs", {}).get("horizontal")) != "3857":
        sys.exit("This survey is not stored on the web-map grid; this script only handles that case.")

    # The web-map grid stretches distances by 1/cos(latitude). Undo that to get real metres.
    stretch = 1.0 / math.cos(math.radians(args.lat))
    cx, cy = to_web_map(args.lon, args.lat)
    half = args.size / 2 * stretch
    box = (cx - half, cy - half, cx + half, cy + half)

    started = time.time()
    files = find_files(root, info, box, args.depth)
    print(f"{len(files)} files to fetch, down to depth {args.depth}", flush=True)

    parts = []
    with ThreadPoolExecutor(args.threads) as pool:
        fetch = lambda f: read_file(root, f[0], box, (cx, cy), stretch)
        for done, part in enumerate(pool.map(fetch, files), 1):
            parts.append(part)
            if done % 200 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files, {sum(len(p[0]) for p in parts):,} dots", flush=True)
    east, north, up, intensity, kind, returns = (np.concatenate(c) for c in zip(*parts))
    del parts

    good = (kind != 7) & (kind != 18)                       # drop the survey's "noise" dots (birds, glitches)
    east, north, up, intensity, kind, returns = (a[good] for a in (east, north, up, intensity, kind, returns))
    if len(east) == 0:
        sys.exit("No dots found there. Check the survey name and the place.")

    # Ground level at the centre: the median height of ground dots within 30 m of it.
    near = (np.abs(east) < 30) & (np.abs(north) < 30) & (kind == 2)
    ground = float(np.median(up[near])) if near.any() else float(np.median(up))

    # Viewer axes: x = east, y = DOWN, z = north, in metres, centre of the square at the origin.
    points = np.empty((len(east), 3), dtype=np.float32)
    points[:, 0] = east
    points[:, 1] = -(up - ground)
    points[:, 2] = north
    colours = colour(intensity, kind, returns)

    with open(args.out, "wb") as f:
        f.write(b"PWSCENE1")
        f.write(np.int32(len(points)).tobytes())
        f.write(points.tobytes())
        f.write(colours.tobytes())

    # Opening camera: stand off to one side, up high, looking back down at the centre.
    bearing = math.radians(args.view_from)
    px, pz = args.view_distance * math.sin(bearing), args.view_distance * math.cos(bearing)
    yaw = math.atan2(-px, -pz)                              # face the centre
    pitch = -math.atan2(args.view_height, args.view_distance)
    with open(args.out + ".pose", "w") as f:
        f.write("%f %f %f %f %f\n" % (px, -args.view_height, pz, yaw, pitch))

    print(f"saved {args.out}: {len(points):,} dots, {args.size:.0f} m square, "
          f"from {(-points[:, 1]).min():.0f} m to {(-points[:, 1]).max():.0f} m above the centre's ground, "
          f"{time.time() - started:.0f} s", flush=True)
    kinds, counts = np.unique(kind, return_counts=True)
    print("survey labels (label: dots):", ", ".join(f"{k}: {c:,}" for k, c in zip(kinds, counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
