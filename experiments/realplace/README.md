# Real place experiment

A detour from the step-by-step project: real aerial lidar in the viewer, then a
pipeline that fills in what the plane could not see. Nothing in `src/`, `viewer/`
or `tests/` is changed by it.

## The pipeline, one script per stage

| Stage | Script | What it does |
| --- | --- | --- |
| 1 | `fetch_place.py` | Fetches a square of USGS 3DEP lidar from the public AWS bucket. Saves the dots, their survey labels, and a confidence of 100 for each (measured) |
| 2 | `make_walls.py` | Finds where a roof stands well above its neighbour, decides where the straight edges are, and drops walls of dots from roof edge to ground. Wall dots get confidence 10 and a plain colour |
| 3 | `make_route.py` | Lays out the cameras: a walk at eye level and two flyover rings. Draws them on a map |
| 4 | `paint_route.py` | For each camera: draws the scene, finds the wall dots this view sees better than any before, has an image model paint only those, and copies the colours onto the dots |

`pipeline.ipynb` runs all four on Colab and then opens the viewer on the result.
`stanford.ipynb` is the earlier, simpler notebook: lidar straight into the viewer.

## The viewer side

| File | What it is |
| --- | --- |
| `kernels_real.h` | `splatKernel` (each dot a small square sized by distance) and `resolveSkyKernel` (empty boxes become sky) |
| `viewer_real.cu` | `src/viewer.cu` with those two kernels, the opening pose read from a file, and the dot size from the command line |

## Confidence

One number per dot, 0 to 100, in the `.conf` file next to each scene:

- 100: measured by the laser. Never repainted.
- 10: a wall dot we made up, still in its plain colour.
- 20 to 70: a wall dot that has been painted. Higher means the camera that painted
  it was closer and looked at the wall more squarely. A dot is only repainted by a
  view that scores higher than the one that painted it last.

## What is assumed, not measured

- Walls are vertical and stand directly under the roof edge. Real roofs overhang, so
  the walls sit a little further out than the real ones.
- Open arcades and passages under a roof become solid walls.
- What a wall looks like is invented by the image model. Only its position and size
  come from the lidar.

Data: USGS 3D Elevation Program, survey `CA_SantaClaraCounty_2020`, public domain.
The survey has no camera colour, so measured dots are coloured by the survey's own
labels and shaded by how strongly each one reflected the laser.
