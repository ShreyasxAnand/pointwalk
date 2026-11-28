# production

Builds a point cloud from Waymo lidar data, on the GPU.

## Pipeline

| Stage | What | File |
| --- | --- | --- |
| 1 | Waymo Parquet -> plain `.bin` files (once per segment) | `waymo/convert.py` |
| 2 | Read one range image and the calibration | `waymo/waymofile.h` |
| 3 | Calibration -> a tilt per row, a turn per column, the mount | `adapters/waymo_adapter.h` |
| 4 | Range image -> one dot per box, in the vehicle frame (GPU) | `generic/kernels.cuh`, `generic/geometry.h` |
| 5 | Drop empty boxes, grey from intensity, write the point cloud | `generic/pointcloud.h` |

Stages 2 to 5 are `adapters/waymo_to_points.cu`, normally started by `run.py`.
The adapter (stage 3) runs on the CPU, once per laser.

## Run

```sh
source ~/venvs/pointwalk/bin/activate

# Stage 1: convert a segment into data/<segment>/ (needs a Google login for gs://)
python production/waymo/convert.py --input gs://waymo_open_dataset_v_2_0_1/training \
    --segment <segment> --out production/data/<segment> [--frames N]

# Stages 2 to 5: list the converted segments, then build one frame
python production/run.py
python production/run.py <segment> [--frame F] [--laser L]
```

- `--frames N` converts frames 0 to N - 1. Frame numbers start at 0.
- `--laser` defaults to 1, the top laser. Lasers 2 to 5 are not supported yet.
- Output: `output/<segment>/frameFFFF_laserL_vehicle.bin`.
- `data/` and `output/` are links to `~/data/waymo` and `~/data/pointwalk_out`.
