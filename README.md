# scene-viz

A learning project: a point-cloud scene visualizer in CUDA. It starts from a
synthetic scene (floor + sphere) rendered to a depth image with a z-buffer on
both CPU and GPU, and grows toward a walkable scene built from aerial lidar,
aerial imagery and 360 panoramas.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | CUDA sources (`.cu`) and shared C++ headers |
| `tests/` | CPU-only tests, compiled locally with `g++` |
| `viewer/` | The viewer's web server (`server.py`) and its page (`index.html`) |
| `notebooks/launcher.ipynb` | One Colab cell: clone or pull, compile, run, display |
| `notebooks/viewer.ipynb` | One Colab cell: compile, start the viewer, show its page |

## Running

GPU code runs on Google Colab (T4), not locally.

1. Commit and push to the `dev` branch.
2. Open `notebooks/launcher.ipynb` in Colab with a T4 GPU runtime.
3. Run the single cell. It pulls `dev`, compiles each program in `src/` and
   runs it from `build/`.

## Walking around

Open `notebooks/viewer.ipynb` in Colab (T4) and run its cell. It compiles
`src/viewer.cu`, starts it behind a small web server, and shows a page: click
the picture, then use W A S D, Q E and the arrow keys. How a key press changes
the camera is `applyMove` in `src/move.h`.

Compile flags: `nvcc -O2 -arch=sm_75 -fmad=false`. `-fmad=false` stops the
compiler from fusing multiply-add, which keeps CPU and GPU results identical
bit for bit.

## Branches

`dev` holds work in progress. `main` only moves when a step's check passes.
