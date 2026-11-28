#ifndef POINTWALK_GENERIC_KERNELS_CUH
#define POINTWALK_GENERIC_KERNELS_CUH

#include "geometry.h"   // rangeMinReturn, SensorToVehicle, sphericalToSensor, sensorToVehicle
#include "point.h"      // Point

// Every CUDA kernel, each with its own comment. Only .cu files include this.

// ===== Step 7: unprojection, range image -> vehicle frame =====

// Range image -> one dot per box, in the vehicle frame (unprojection).
// In:  the distance grid (H x W), a tilt per row, a turn per column, the mount.
// Out: one dot per box, same order. Empty boxes become (0, 0, 0).
// One thread per box, each writes only its own slot: no atomics.
__global__ void rangeToVehicleKernel(const float* ranges, int H, int W,
                                     const float* tiltOfRow, const float* turnOfColumn,
                                     SensorToVehicle mount, Point* points) {
  const int box = blockIdx.x * blockDim.x + threadIdx.x;
  if (box >= H * W) return;                                // the launch rounds up: extra thread
  const float range = ranges[box];
  if (range <= rangeMinReturn) {
    points[box] = Point{0.0f, 0.0f, 0.0f};
    return;
  }
  const int row = box / W;
  const int col = box % W;
  const float tilt = tiltOfRow[row];
  const float turn = turnOfColumn[col];
  float xs, ys, zs;
  sphericalToSensor(tilt, turn, range, xs, ys, zs);
  float xv, yv, zv;
  sensorToVehicle(xs, ys, zs, mount, xv, yv, zv);
  points[box] = Point{xv, yv, zv};
}

#endif
