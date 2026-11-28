// GPU test: rangeToVehicleKernel (production/generic/kernels.cuh) on a dirty buffer.
// The output buffer is filled with 0xFF bytes first, as a reused buffer would
// be. Every box must be written: empty boxes as (0, 0, 0), the others as the
// CPU math gives them (within 0.1 mm, the step 7 rule for sinf/cosf rounding).
//
// Needs a GPU. From the repo root:
//   nvcc -O2 -arch=sm_75 -fmad=false tests/test_range_to_vehicle_kernel.cu \
//        -o build/test_range_to_vehicle_kernel && ./build/test_range_to_vehicle_kernel

#include <cmath>
#include <cstdio>
#include <vector>
#include "../production/generic/kernels.cuh"   // rangeToVehicleKernel
#include "../production/generic/launch.cuh"    // CHECK

int failures = 0;   // how many checks have failed so far

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
  printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) failures++;
}

int main() {
  const int H = 2, W = 3, N = H * W;
  const std::vector<float> ranges = {5.0f, -1.0f, 0.0f, 2.0f, -1.0f, 3.0f};   // boxes 1, 2, 4 empty
  const std::vector<float> tiltOfRow = {0.0f, 0.1f};
  const std::vector<float> turnOfColumn = {0.0f, 0.2f, 0.4f};
  const SensorToVehicle mount = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 0, 0}};

  float* d_ranges;
  float* d_tiltOfRow;
  float* d_turnOfColumn;
  Point* d_points;
  CHECK(cudaMalloc(&d_ranges, N * sizeof(float)));
  CHECK(cudaMalloc(&d_tiltOfRow, H * sizeof(float)));
  CHECK(cudaMalloc(&d_turnOfColumn, W * sizeof(float)));
  CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
  CHECK(cudaMemcpy(d_ranges, ranges.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_tiltOfRow, tiltOfRow.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_turnOfColumn, turnOfColumn.data(), W * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_points, 0xFF, N * sizeof(Point)));   // garbage, as a reused buffer would hold

  rangeToVehicleKernel<<<1, 256>>>(d_ranges, H, W, d_tiltOfRow, d_turnOfColumn, mount, d_points);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<Point> p(N);
  CHECK(cudaMemcpy(p.data(), d_points, N * sizeof(Point), cudaMemcpyDeviceToHost));
  cudaFree(d_ranges);
  cudaFree(d_tiltOfRow);
  cudaFree(d_turnOfColumn);
  cudaFree(d_points);

  char name[200];
  for (int box = 0; box < N; box++) {
    const int row = box / W, col = box % W;
    printf("      box %d range %4.1f -> (%g, %g, %g)\n", box, ranges[box], p[box].x, p[box].y, p[box].z);
    if (ranges[box] <= rangeMinReturn) {
      snprintf(name, sizeof name, "%d. box %d (range %g) is empty: written as (0, 0, 0)", box + 1, box, ranges[box]);
      report(name, p[box].x == 0.0f && p[box].y == 0.0f && p[box].z == 0.0f);
    } else {
      float xs, ys, zs, xv, yv, zv;   // the same box on the CPU
      sphericalToSensor(tiltOfRow[row], turnOfColumn[col], ranges[box], xs, ys, zs);
      sensorToVehicle(xs, ys, zs, mount, xv, yv, zv);
      const float off = fmaxf(fabsf(p[box].x - xv), fmaxf(fabsf(p[box].y - yv), fabsf(p[box].z - zv)));
      snprintf(name, sizeof name, "%d. box %d (range %g) matches the CPU within 0.1 mm (off by %g m)",
               box + 1, box, ranges[box], off);
      report(name, off < 0.0001f);
    }
  }
  report("7. box 0 (straight ahead, 5 m) is exactly (0, 0, 5)",
         p[0].x == 0.0f && p[0].y == 0.0f && p[0].z == 5.0f);

  printf("\n%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
