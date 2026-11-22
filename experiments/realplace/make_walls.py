#!/usr/bin/env python3
"""Stage 2: add the walls the plane could not see.

The laser looked straight down, so it measured roofs and ground but no walls.
This script finds every place where a roof stands well above what is next to it,
decides where the straight edges are, and drops a wall of dots from the roof edge
down to the lower surface.

    python3 make_walls.py --scene quad.bin --out quad_walls.bin

Reads   quad.bin, quad.bin.labels, quad.bin.conf (and quad.bin.pose)
Writes  quad_walls.bin plus .labels, .conf, .normals and .pose

Needs:  pip install numpy scipy
"""
import argparse
import shutil

import numpy as np
from scipy import ndimage

CELL = 0.5          # metres. The top-down grid the roofs are examined on.
MIN_DROP = 2.0      # metres. A roof this much higher than its neighbour gets a wall.
MIN_BUILDING = 40   # grid cells (10 square metres). Smaller blobs are ignored.
LINE_TOLERANCE = 0.4    # metres. How far an edge point may sit from a line and still belong to it.
LINE_GAP = 1.5          # metres. A longer gap than this ends a wall.
LINE_MIN_POINTS = 8     # a straight wall needs at least this many edge points
LINE_REACH = 12.0       # metres. How far away the second point of a pair may be.
LINE_TRIES = 25         # nearby partners tried for each starting point
DOT_STEP = 0.12         # metres between wall dots, along the wall and up it
WALL_LABEL = 64         # our own label for "wall dot, not measured"
WALL_CONFIDENCE = 10    # out of 100: the shape is an assumption and the colour is a placeholder
WALL_COLOUR = np.array([214, 190, 150], dtype=np.float32)   # plain sandstone until it is painted
SUN = np.array([-0.6, -0.8])    # (east, north) direction the light comes from, for simple shading

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
rng = np.random.default_rng(0)

# ---- Read the scene: 8 letters, the dot count, the positions, the colours ----
with open(args.scene, "rb") as f:
    assert f.read(8) == b"PWSCENE1", "not a scene file"
    count = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    points = np.frombuffer(f.read(count * 12), dtype=np.float32).reshape(count, 3)
    colours = np.frombuffer(f.read(count * 3), dtype=np.uint8).reshape(count, 3)
labels = np.fromfile(args.scene + ".labels", dtype=np.uint8)
confidence = np.fromfile(args.scene + ".conf", dtype=np.uint8)

east, up, north = points[:, 0], -points[:, 1], points[:, 2]    # the viewer's y points down

# ---- Lay a top-down grid over the scene ----
east0, north0 = east.min(), north.min()
cols = int((east.max() - east0) / CELL) + 1
rows = int((north.max() - north0) / CELL) + 1
cell = ((north - north0) / CELL).astype(np.int64) * cols + ((east - east0) / CELL).astype(np.int64)

# Roof height: the highest building dot in each cell.
roof = np.full(rows * cols, -np.inf, dtype=np.float32)
is_roof_dot = labels == 6
np.maximum.at(roof, cell[is_roof_dot], up[is_roof_dot])
roof = roof.reshape(rows, cols)

# Ground height: the lowest ground dot in each cell. Cells with none (under roofs
# and trees) copy the nearest cell that has one.
ground = np.full(rows * cols, np.inf, dtype=np.float32)
is_ground_dot = labels == 2
np.minimum.at(ground, cell[is_ground_dot], up[is_ground_dot])
ground = ground.reshape(rows, cols)
nearest = ndimage.distance_transform_edt(np.isinf(ground), return_distances=False, return_indices=True)
ground = ground[nearest[0], nearest[1]]

# Buildings: connected blobs of roof cells, ignoring tiny ones.
blobs, blob_count = ndimage.label(np.isfinite(roof))
sizes = np.bincount(blobs.ravel())
blobs[sizes[blobs] < MIN_BUILDING] = 0
is_building = blobs > 0

# The height of whatever the plane saw in each cell: the roof, or else the ground.
surface = np.where(is_building, roof, ground)

# ---- Find the edge points: roof cells that stand MIN_DROP above a neighbour ----
# Each edge point remembers: where it is, how high the roof is there (top), the
# height of the lower neighbour (bottom), which way is "out", and its building.
edge_east, edge_north, edge_top, edge_bottom, edge_out, edge_blob = [], [], [], [], [], []
for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
    neighbour = np.roll(surface, (-d_row, -d_col), axis=(0, 1))    # the surface one cell over
    drop = np.where(is_building, roof - neighbour, 0.0)
    # np.roll wraps around the border, so the outermost ring of cells is left out.
    drop[0, :] = drop[-1, :] = 0
    drop[:, 0] = drop[:, -1] = 0
    r, c = np.nonzero(drop > MIN_DROP)
    edge_east.append(east0 + (c + 0.5 + 0.5 * d_col) * CELL)       # the middle of the shared cell side
    edge_north.append(north0 + (r + 0.5 + 0.5 * d_row) * CELL)
    edge_top.append(roof[r, c])
    edge_bottom.append(neighbour[r, c])
    edge_out.append(np.tile(np.array([d_col, d_row], dtype=np.float32), (len(r), 1)))
    edge_blob.append(blobs[r, c])
edge_xy = np.stack([np.concatenate(edge_east), np.concatenate(edge_north)], axis=1)
edge_top, edge_bottom = np.concatenate(edge_top), np.concatenate(edge_bottom)
edge_out, edge_blob = np.concatenate(edge_out), np.concatenate(edge_blob)
print(f"{blob_count} roof blobs, {int(is_building.sum() * CELL * CELL):,} square metres of roof, "
      f"{len(edge_xy):,} edge points", flush=True)

# ---- The straight line decider ----
# For one building at a time: take an edge point that has no wall yet. Pair it with a
# few nearby edge points in turn. Each pair gives a line; count the edge points that
# sit on that line in one unbroken run through the first point. Keep the best run.
# If it is long enough it becomes a wall and its points are used up. If not, the
# first point is set aside as "loose". Repeat until every edge point is dealt with.
wall_points, wall_normals = [], []
straight_walls = loose_points = 0

for blob in np.unique(edge_blob):
    mine = np.nonzero(edge_blob == blob)[0]
    xy, top, bottom, out = edge_xy[mine], edge_top[mine], edge_bottom[mine], edge_out[mine]
    left = np.ones(len(mine), dtype=bool)          # edge points not yet dealt with
    loose = np.zeros(len(mine), dtype=bool)        # edge points that fit no straight wall

    while left.any():
        a = rng.choice(np.nonzero(left)[0])
        offset = xy - xy[a]
        distance = np.hypot(offset[:, 0], offset[:, 1])
        partners = np.nonzero(left & (distance > 1.5) & (distance < LINE_REACH))[0]
        best = np.array([a])
        for b in rng.permutation(partners)[:LINE_TRIES]:
            along_dir = offset[b] / distance[b]
            across_dir = np.array([-along_dir[1], along_dir[0]])
            on_line = np.nonzero(left & (np.abs(offset @ across_dir) < LINE_TOLERANCE))[0]
            # Keep only the unbroken run of points that contains point a (which sits at position 0).
            order = on_line[np.argsort(offset[on_line] @ along_dir)]
            position = offset[order] @ along_dir
            breaks = np.nonzero(np.diff(position) > LINE_GAP)[0] + 1
            starts = np.concatenate([[0], breaks])
            ends = np.concatenate([breaks, [len(order)]])
            run = np.searchsorted(ends, np.searchsorted(position, 0.0), side="right")
            members = order[starts[run]:ends[run]]
            if len(members) > len(best):
                best = members
        if len(best) < LINE_MIN_POINTS:
            left[a] = False
            loose[a] = True
            continue

        # Refit the line through all the points it won (the two random ones were only a first guess).
        centre = xy[best].mean(axis=0)
        _, _, axes = np.linalg.svd(xy[best] - centre)
        along_dir = axes[0]
        normal = np.array([-along_dir[1], along_dir[0]])
        if (out[best] @ normal).sum() < 0:          # make the normal point out, toward the lower side
            normal = -normal
        position = (xy[best] - centre) @ along_dir
        order = np.argsort(position)
        position, wall_top, wall_bottom = position[order], top[best][order], bottom[best][order]
        # Smooth the roof line and the base line along the wall, so one odd cell does not make a spike.
        wall_top = ndimage.median_filter(wall_top, size=7, mode="nearest")
        wall_bottom = ndimage.median_filter(wall_bottom, size=7, mode="nearest")

        # One column of dots every DOT_STEP along the wall, from the base up to the roof line.
        steps = np.arange(position[0], position[-1] + DOT_STEP, DOT_STEP)
        column_top = np.interp(steps, position, wall_top)
        column_bottom = np.interp(steps, position, wall_bottom)
        dots_in_column = np.maximum(np.ceil((column_top - column_bottom) / DOT_STEP), 0).astype(np.int64)
        column = np.repeat(np.arange(len(steps)), dots_in_column)
        rung = np.arange(dots_in_column.sum()) - np.repeat(np.cumsum(dots_in_column) - dots_in_column, dots_in_column)
        dot_xy = centre + steps[column, None] * along_dir
        dot_up = column_bottom[column] + rung * DOT_STEP
        wall_points.append(np.stack([dot_xy[:, 0], -dot_up, dot_xy[:, 1]], axis=1))
        wall_normals.append(np.tile(normal, (len(column), 1)))
        straight_walls += 1
        left[best] = False

    # Edge points that fit no straight line (curved walls, odd corners): each gets its own
    # column of dots, so there is still no hole, just no straightening.
    for i in np.nonzero(loose)[0]:
        heights = np.arange(bottom[i], top[i], DOT_STEP)
        side = np.array([-out[i][1], out[i][0]])    # the direction along the cell side
        for shift in np.arange(-CELL / 2, CELL / 2, DOT_STEP):
            spot = xy[i] + shift * side
            wall_points.append(np.stack([np.full(len(heights), spot[0]), -heights,
                                         np.full(len(heights), spot[1])], axis=1))
            wall_normals.append(np.tile(out[i], (len(heights), 1)))
        loose_points += 1

wall_points = np.concatenate(wall_points).astype(np.float32)
wall_normals = np.concatenate(wall_normals).astype(np.float32)

# Plain colour, a little darker on the sides facing away from the light, so the shapes read.
facing = np.clip(wall_normals @ (-SUN / np.hypot(*SUN)), 0.0, 1.0)
shade = 0.72 + 0.28 * facing + rng.uniform(-0.03, 0.03, len(wall_points))
wall_colours = np.clip(WALL_COLOUR * shade[:, None], 0, 255).astype(np.uint8)

print(f"{straight_walls:,} straight walls, {loose_points:,} edge points left as loose columns, "
      f"{len(wall_points):,} wall dots", flush=True)

# ---- Write the new scene: the measured dots first, then the wall dots ----
all_points = np.concatenate([points, wall_points])
with open(args.out, "wb") as f:
    f.write(b"PWSCENE1")
    f.write(np.int32(len(all_points)).tobytes())
    f.write(all_points.tobytes())
    f.write(np.concatenate([colours, wall_colours]).tobytes())
np.concatenate([labels, np.full(len(wall_points), WALL_LABEL, np.uint8)]).tofile(args.out + ".labels")
np.concatenate([confidence, np.full(len(wall_points), WALL_CONFIDENCE, np.uint8)]).tofile(args.out + ".conf")
# .normals: which way each dot's surface faces, as (east, north). Zero for measured dots.
np.concatenate([np.zeros((count, 2), np.float32), wall_normals]).tofile(args.out + ".normals")
# .ground: the ground height map, so the route can stand a person on the ground.
np.savez(args.out + ".ground.npz", ground=ground, building=is_building,
         east0=east0, north0=north0, cell=CELL)
shutil.copy(args.scene + ".pose", args.out + ".pose")
print(f"saved {args.out}: {len(all_points):,} dots ({count:,} measured + {len(wall_points):,} wall)")
