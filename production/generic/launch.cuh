#ifndef POINTWALK_GENERIC_LAUNCH_CUH
#define POINTWALK_GENERIC_LAUNCH_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels.cuh"   // rangeToVehicleKernel
#include "point.h"       // Point

// Host functions that run a kernel: allocate, copy in, launch, copy back, free.
// Only .cu files include this.

// Exit with a message if a CUDA call fails (CUDA errors are otherwise silent).
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ===== Step 7: unprojection, range image -> vehicle frame =====

// Range image -> one dot per box, in the vehicle frame, on the GPU.
// In:  ranges (H x W), a tilt per row, a turn per column, the mount.
// Out: one Point per box, same order. Empty boxes become (0, 0, 0).
inline std::vector<Point> rangeToVehicleGPU(const std::vector<float>& ranges, int H, int W,
                                            const std::vector<float>& tiltOfRow,
                                            const std::vector<float>& turnOfColumn,
                                            SensorToVehicle mount) {
  const int numBoxes = H * W;
  float* d_ranges;
  float* d_tiltOfRow;
  float* d_turnOfColumn;
  Point* d_points;
  CHECK(cudaMalloc(&d_ranges, numBoxes * sizeof(float)));
  CHECK(cudaMalloc(&d_tiltOfRow, H * sizeof(float)));
  CHECK(cudaMalloc(&d_turnOfColumn, W * sizeof(float)));
  CHECK(cudaMalloc(&d_points, numBoxes * sizeof(Point)));
  CHECK(cudaMemcpy(d_ranges, ranges.data(), numBoxes * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_tiltOfRow, tiltOfRow.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_turnOfColumn, turnOfColumn.data(), W * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (numBoxes + threads - 1) / threads;   // round up to cover every box
  rangeToVehicleKernel<<<blocks, threads>>>(d_ranges, H, W, d_tiltOfRow, d_turnOfColumn, mount, d_points);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<Point> boxPoints(numBoxes);
  CHECK(cudaMemcpy(boxPoints.data(), d_points, numBoxes * sizeof(Point), cudaMemcpyDeviceToHost));
  cudaFree(d_ranges);
  cudaFree(d_tiltOfRow);
  cudaFree(d_turnOfColumn);
  cudaFree(d_points);
  return boxPoints;
}

#endif
