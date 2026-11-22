#!/usr/bin/env python3
"""Stage 3: decide where the cameras stand.

Two kinds of stop:
  walk     a person at eye level following WALK, stopping every few metres and
           looking ahead, to the left and to the right
  flyover  two rings around the middle, one high and one low, looking inward

Writes route.txt (one camera per line: x y z yaw pitch kind) and route.png, a map
with every stop drawn on it, so the route can be checked before any painting.

    python3 make_route.py --scene quad_walls.bin --out route.txt

Needs:  pip install numpy pillow
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw

# The walk, as corner points in metres: (east, north) measured from the middle of the
# scene. Edit these to change the route. This one comes in from the north through
# Memorial Court, then loops around the inside of Stanford's Main Quad.
WALK = [(5, 185), (-5, 120), (-12, 60), (-25, 0), (-85, 12), (-60, -20),
        (-38, -25), (30, -30), (45, 5), (-25, 0)]
STOP_EVERY = 8.0          # metres between stops along the walk
EYE_HEIGHT = 1.7          # metres above the ground
LOOK_SIDEWAYS = 75.0      # degrees to the left and right of the walking direction
KEEP_CLEAR = 2.0          # metres. A stop closer than this to a building is skipped.

RING_CENTRE = (-25, 0)    # (east, north) the flyover rings look at
RINGS = [(170.0, 70.0, 12),   # (radius in metres, height above the ground, number of stops)
         (110.0, 25.0, 12)]

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--out", default="route.txt")
args = parser.parse_args()

# The ground height map and the building map that make_walls.py saved.
saved = np.load(args.scene + ".ground.npz")
ground, building = saved["ground"], saved["building"]
east0, north0, cell = float(saved["east0"]), float(saved["north0"]), float(saved["cell"])
rows, cols = ground.shape

# How far each grid cell is from the nearest building, in metres.
from scipy import ndimage
clearance = ndimage.distance_transform_edt(~building) * cell


def grid(east, north):
    """Which grid cell a spot is in, clamped to the map."""
    return (min(max(int((north - north0) / cell), 0), rows - 1),
            min(max(int((east - east0) / cell), 0), cols - 1))


cameras = []       # (east, height above scene zero, north, yaw, pitch, kind)

# ---- The walk ----
skipped = 0
for (e1, n1), (e2, n2) in zip(WALK[:-1], WALK[1:]):
    length = math.hypot(e2 - e1, n2 - n1)
    heading = math.atan2(e2 - e1, n2 - n1)            # yaw: 0 faces north, positive turns toward east
    for step in np.arange(0.0, length, STOP_EVERY):
        east = e1 + (e2 - e1) * step / length
        north = n1 + (n2 - n1) * step / length
        r, c = grid(east, north)
        if clearance[r, c] < KEEP_CLEAR:
            skipped += 1
            continue
        height = ground[r, c] + EYE_HEIGHT
        for turn in (0.0, -LOOK_SIDEWAYS, LOOK_SIDEWAYS):
            cameras.append((east, height, north, heading + math.radians(turn), 0.0, "walk"))

# ---- The flyover ----
for radius, above, stops in RINGS:
    for k in range(stops):
        angle = 2 * math.pi * k / stops
        east = RING_CENTRE[0] + radius * math.sin(angle)
        north = RING_CENTRE[1] + radius * math.cos(angle)
        r, c = grid(*RING_CENTRE)
        height = ground[r, c] + above
        yaw = math.atan2(RING_CENTRE[0] - east, RING_CENTRE[1] - north)   # face the centre
        pitch = -math.atan2(above, radius)                                 # tilt down toward it
        cameras.append((east, height, north, yaw, pitch, "flyover"))

# ---- Save: the viewer's axes are x = east, y = DOWN, z = north ----
with open(args.out, "w") as f:
    for east, height, north, yaw, pitch, kind in cameras:
        f.write("%.3f %.3f %.3f %.5f %.5f %s\n" % (east, -height, north, yaw, pitch, kind))

# ---- Draw the map: buildings in orange, each stop as a dot with a short line showing where it looks ----
picture = np.zeros((rows, cols, 3), dtype=np.uint8)
picture[:] = np.clip((ground - ground.min()) / max(float(np.ptp(ground)), 1.0) * 110 + 60, 0, 255)[:, :, None]
picture[building] = (214, 150, 90)
image = Image.fromarray(picture[::-1])                 # north at the top
draw = ImageDraw.Draw(image)
for east, height, north, yaw, pitch, kind in cameras:
    px = (east - east0) / cell
    py = rows - (north - north0) / cell
    colour = (60, 220, 255) if kind == "walk" else (255, 80, 200)
    reach = 12 if kind == "walk" else 24
    draw.line([px, py, px + reach * math.sin(yaw), py - reach * math.cos(yaw)], fill=colour, width=1)
    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=colour)
image.save(args.out.rsplit(".", 1)[0] + ".png")

walk = sum(1 for c in cameras if c[5] == "walk")
print(f"{len(cameras)} cameras: {walk} on the walk ({walk // 3} stops, {skipped} skipped for being "
      f"too close to a building), {len(cameras) - walk} on the flyover")
print("saved", args.out, "and", args.out.rsplit(".", 1)[0] + ".png")
