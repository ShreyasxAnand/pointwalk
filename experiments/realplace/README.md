# Real place experiment

One night's detour from the step-by-step project: real aerial lidar in the viewer.
Nothing in `src/`, `viewer/` or `tests/` is changed by it.

| File | What it is |
| --- | --- |
| `fetch_place.py` | Fetches a square of USGS 3DEP lidar from the public AWS bucket and saves it as a scene file plus an opening camera pose |
| `kernels_real.h` | `splatKernel`: the first pass, painting each dot as a small square sized by distance |
| `viewer_real.cu` | `src/viewer.cu` with three changes: opening pose from a file, `splatKernel`, dot size from the command line |
| `stanford.ipynb` | One Colab cell: compile, fetch Stanford's Main Quad, start the viewer, show the page |

Data: USGS 3D Elevation Program, survey `CA_SantaClaraCounty_2020`, public domain.
The survey has no camera colour, so dots are coloured by the survey's own labels
(ground, plants, buildings, water) and shaded by how strongly each one reflected the laser.
