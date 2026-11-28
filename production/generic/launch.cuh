#ifndef POINTWALK_GENERIC_LAUNCH_CUH
#define POINTWALK_GENERIC_LAUNCH_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels.cuh"       // rangeToVehicleKernel
#include "point.h"           // Point
#include "range_image.h"     // SensorModel, RangeFrame

// Host functions that run a kernel: allocate, copy in, launch, copy back, free.
// Only .cu files include this.

// Exit with a message if a CUDA call fails (CUDA errors are otherwise silent).
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (%s line %d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ===== Step 7: unprojection, range image -> vehicle frame =====

// Range image -> one dot per box, in the vehicle frame, on the GPU.
// In:  the sensor (H x W, tilts, turns, mount), one frame (H x W ranges and colours).
// Out: boxPoints, one Point per box, same order. Empty boxes become (0, 0, 0).
//      false, with a message, if a size is wrong; boxPoints is then not changed.
inline bool rangeToVehicleGPU(const SensorModel& sensor, const RangeFrame& frame,
                              std::vector<Point>& boxPoints) {
  const int H = sensor.H, W = sensor.W;
  if (H <= 0 || W <= 0) {
    fprintf(stderr, "rangeToVehicleGPU: sensor is %d x %d, both must be at least 1\n", H, W);
    return false;
  }
  if (frame.H != H || frame.W != W) {
    fprintf(stderr, "rangeToVehicleGPU: sensor is %d x %d but the frame is %d x %d\n",
            H, W, frame.H, frame.W);
    return false;
  }
  const size_t numBoxes = (size_t)H * W;
  if (sensor.tiltOfRow.size() != (size_t)H) {
    fprintf(stderr, "rangeToVehicleGPU: tiltOfRow has %zu entries, H is %d\n", sensor.tiltOfRow.size(), H);
    return false;
  }
  if (sensor.turnOfColumn.size() != (size_t)W) {
    fprintf(stderr, "rangeToVehicleGPU: turnOfColumn has %zu entries, W is %d\n", sensor.turnOfColumn.size(), W);
    return false;
  }
  if (frame.ranges.size() != numBoxes) {
    fprintf(stderr, "rangeToVehicleGPU: ranges has %zu entries, H x W is %zu\n", frame.ranges.size(), numBoxes);
    return false;
  }
  if (frame.colors.size() != numBoxes) {
    fprintf(stderr, "rangeToVehicleGPU: colors has %zu entries, H x W is %zu\n", frame.colors.size(), numBoxes);
    return false;
  }

  float* d_ranges;
  float* d_tiltOfRow;
  float* d_turnOfColumn;
  Point* d_points;
  CHECK(cudaMalloc(&d_ranges, numBoxes * sizeof(float)));
  CHECK(cudaMalloc(&d_tiltOfRow, H * sizeof(float)));
  CHECK(cudaMalloc(&d_turnOfColumn, W * sizeof(float)));
  CHECK(cudaMalloc(&d_points, numBoxes * sizeof(Point)));
  CHECK(cudaMemcpy(d_ranges, frame.ranges.data(), numBoxes * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_tiltOfRow, sensor.tiltOfRow.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_turnOfColumn, sensor.turnOfColumn.data(), W * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (int)((numBoxes + threads - 1) / threads);   // round up to cover every box
  rangeToVehicleKernel<<<blocks, threads>>>(d_ranges, H, W, d_tiltOfRow, d_turnOfColumn, sensor.mount, d_points);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  boxPoints.resize(numBoxes);
  CHECK(cudaMemcpy(boxPoints.data(), d_points, numBoxes * sizeof(Point), cudaMemcpyDeviceToHost));
  cudaFree(d_ranges);
  cudaFree(d_tiltOfRow);
  cudaFree(d_turnOfColumn);
  cudaFree(d_points);
  return true;
}

#endif
