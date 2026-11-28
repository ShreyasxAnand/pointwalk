// GPU test: rangeToVehicleGPU (production/generic/launch.cuh) refuses wrong sizes.
// A good sensor and frame (2 x 3) must work; each bad size must return false
// before the GPU is touched, print its message, and leave the output alone.
//
// Needs a GPU. From the repo root:
//   nvcc -O2 -arch=sm_75 -fmad=false tests/test_range_to_vehicle_gpu.cu \
//        -o build/test_range_to_vehicle_gpu && ./build/test_range_to_vehicle_gpu

#include <cstdio>
#include <vector>
#include "../production/generic/launch.cuh"   // rangeToVehicleGPU

int failures = 0;   // how many checks have failed so far

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
  printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) failures++;
}

// A bad case: must return false and leave out exactly as it was.
void reportRefused(const char* name, const SensorModel& sensor, const RangeFrame& frame) {
  std::vector<Point> out = {Point{7.0f, 8.0f, 9.0f}};   // a marker that must survive
  fflush(stdout);                                        // keep the message next to its check
  const bool ok = rangeToVehicleGPU(sensor, frame, out);
  fflush(stderr);
  const bool untouched = out.size() == 1 && out[0].x == 7.0f && out[0].y == 8.0f && out[0].z == 9.0f;
  report(name, !ok && untouched);
}

int main() {
  SensorModel sensor;
  sensor.H = 2;
  sensor.W = 3;
  sensor.tiltOfRow.assign(2, 0.0f);
  sensor.turnOfColumn.assign(3, 0.0f);
  sensor.mount = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 0, 0}};
  RangeFrame frame;
  frame.H = 2;
  frame.W = 3;
  frame.ranges.assign(6, 1.0f);
  frame.colors.assign(6, Color{0, 0, 0});

  std::vector<Point> out;
  const bool ok = rangeToVehicleGPU(sensor, frame, out);
  report("1. good sizes: returns true, one dot per box (6)", ok && out.size() == 6);

  RangeFrame wideFrame = frame;
  wideFrame.W = 4;
  reportRefused("2. frame 2 x 4, sensor 2 x 3: refused", sensor, wideFrame);

  SensorModel longTilts = sensor;
  longTilts.tiltOfRow.resize(5);
  reportRefused("3. 5 tilts for 2 rows: refused", longTilts, frame);

  SensorModel noTurns = sensor;
  noTurns.turnOfColumn.clear();
  reportRefused("4. 0 turns for 3 columns: refused", noTurns, frame);

  RangeFrame shortRanges = frame;
  shortRanges.ranges.resize(5);
  reportRefused("5. 5 ranges for 6 boxes: refused", sensor, shortRanges);

  RangeFrame longColors = frame;
  longColors.colors.resize(7);
  reportRefused("6. 7 colours for 6 boxes: refused", sensor, longColors);

  SensorModel noRows = sensor;
  noRows.H = 0;
  reportRefused("7. sensor with 0 rows: refused", noRows, frame);

  printf("\n%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
