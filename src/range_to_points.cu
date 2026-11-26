#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "image.h"         // Color
#include "point.h"         // Point
#include "rangepoints.h"   // panoUnproject, sensorToScene, rangeToPointsCPU, rangeMinReturn
#include "scenefile.h"     // writeScene: saves the dots so the viewer can show them
#include "../io/waymo/waymofile.h"       // readWaymoRange, readWaymoCalibration (raw)
#include "../io/waymo/waymo_adapter.h"   // raw Waymo numbers -> the kernel's plain inputs

// Step 7: one range image -> dots in the vehicle frame, on the GPU, checked
// against the CPU version in rangepoints.h.
//
// Run:  ./range_to_points <converted folder> [frame] [laser]
//   <converted folder>  the --out folder of waymo/convert.py
//   frame               frame number, 0 if left out
//   laser               Waymo's laser name, 1 (the top laser) if left out
//
// SCAFFOLD: main() is complete. The kernel body is empty, and so are the
// functions it and main() call in rangepoints.h and waymo_adapter.h. Those are
// yours.

// rangeToPointsKernel: the GPU version of rangeToPointsCPU.
//
// The big difference from panoDepthKernel: there, one thread = one DOT, and
// many dots could land in the same box. Here, one thread = one BOX, and each
// box makes at most one dot.
//
// Same six jobs as the CPU version (see rangepoints.h). It knows nothing about
// Waymo: it gets the plain arrays described at the top of rangepoints.h.
//
// INPUTS
//   ranges         H * W distances, the box at (row, col) is ranges[row * W + col]
//   tiltOfRow      H angles
//   turnOfColumn   W angles
//   place          where the sensor sits in the scene
//
// OUTPUT
//   points         H * W dots, one per box, same order as ranges. main() has
//                  filled it with zeros.
//
// DECIDE BEFORE YOU WRITE
//   a. How many threads does main() start, and which of them have no box?
//   b. panoDepthKernel needed atomicMinFloat. Does this kernel need anything
//      atomic? Say why before you write it.
//   c. A box with no return (use rangeMinReturn, the named number): leave in
//      points[box] exactly what your CPU version leaves there.
__global__ void rangeToPointsKernel(const float* ranges, int H, int W,
                                    const float* tiltOfRow, const float* turnOfColumn,
                                    SensorToScene place, Point* points) {
  // yours
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    printf("usage: %s <converted folder> [frame] [laser]\n", argv[0]);
    return 1;
  }
  const char* folder = argv[1];
  const int frame = (argc > 2) ? atoi(argv[2]) : 0;
  const int laser = (argc > 3) ? atoi(argv[3]) : 1;

  // ---- Load one range image and the calibration, raw ----
  char rangePath[1024], calibrationPath[1024];
  snprintf(rangePath, sizeof rangePath, "%s/range/frame%04d_laser%d.bin", folder, frame, laser);
  snprintf(calibrationPath, sizeof calibrationPath, "%s/calibration.bin", folder);

  WaymoRangeImage image;
  std::vector<WaymoLaserCalibration> calibrations;
  if (!readWaymoRange(rangePath, image)) return 1;                    // the reader already said why
  if (!readWaymoCalibration(calibrationPath, calibrations)) return 1;

  // Find this laser's calibration by its name, not by its place in the file.
  const WaymoLaserCalibration* cal = nullptr;
  for (const auto& c : calibrations)
    if (c.laser == laser) cal = &c;
  if (!cal) {
    printf("no calibration for laser %d in %s\n", laser, calibrationPath);
    return 1;
  }

  const int H = image.H, W = image.W;
  const int numBoxes = H * W;
  printf("frame %d, laser %d: %d rows x %d columns = %d boxes\n", frame, laser, H, W, numBoxes);

  // ---- The plain range image: channel 0 of every box, same order ----
  std::vector<float> ranges(numBoxes);
  for (int box = 0; box < numBoxes; box++) ranges[box] = image.values[(size_t)box * image.C];

  int numReturns = 0;
  for (int box = 0; box < numBoxes; box++)
    if (ranges[box] > rangeMinReturn) numReturns++;
  printf("boxes with a return: %d of %d\n", numReturns, numBoxes);

  // ---- The adapter (yours): Waymo's calibration -> two angle lists and a placement ----
  std::vector<float> tiltOfRow, turnOfColumn;
  SensorToScene place = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};   // "not moved" until the adapter fills it
  waymoTiltOfRow(*cal, H, tiltOfRow);
  waymoTurnOfColumn(*cal, W, turnOfColumn);
  waymoSensorToVehicle(*cal, place);
  if ((int)tiltOfRow.size() != H || (int)turnOfColumn.size() != W) {
    printf("the adapter gave %d tilts for %d rows and %d turns for %d columns.\n"
           "Both lists must be full: write waymoTiltOfRow and waymoTurnOfColumn in\n"
           "io/waymo/waymo_adapter.h first.\n",
           (int)tiltOfRow.size(), H, (int)turnOfColumn.size(), W);
    return 1;
  }

  // ---- CPU dots ----
  std::vector<Point> cpuPoints(numBoxes, Point{0.0f, 0.0f, 0.0f});
  rangeToPointsCPU(ranges, H, W, tiltOfRow, turnOfColumn, place, cpuPoints);

  // ---- GPU dots ----
  float* d_ranges;
  float* d_tiltOfRow;
  float* d_turnOfColumn;
  Point* d_points;
  cudaMalloc(&d_ranges, numBoxes * sizeof(float));
  cudaMalloc(&d_tiltOfRow, H * sizeof(float));
  cudaMalloc(&d_turnOfColumn, W * sizeof(float));
  cudaMalloc(&d_points, numBoxes * sizeof(Point));

  std::vector<Point> gpuPoints(numBoxes, Point{0.0f, 0.0f, 0.0f});   // starting values: zeros
  cudaMemcpy(d_ranges, ranges.data(), numBoxes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tiltOfRow, tiltOfRow.data(), H * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_turnOfColumn, turnOfColumn.data(), W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points, gpuPoints.data(), numBoxes * sizeof(Point), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (numBoxes + threads - 1) / threads;   // enough blocks to cover every box
  rangeToPointsKernel<<<blocks, threads>>>(d_ranges, H, W, d_tiltOfRow, d_turnOfColumn, place, d_points);
  cudaDeviceSynchronize();
  printf("kernel status: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(gpuPoints.data(), d_points, numBoxes * sizeof(Point), cudaMemcpyDeviceToHost);

  cudaFree(d_ranges);
  cudaFree(d_tiltOfRow);
  cudaFree(d_turnOfColumn);
  cudaFree(d_points);

  // ---- Compare the two sets of dots, box by box (only boxes with a return) ----
  int differ = 0;              // boxes where x, y or z is not identical
  float largestGap = 0.0f;     // the biggest difference in any one of x, y, z, in metres
  int cpuAtZero = 0, gpuAtZero = 0;   // dots still sitting at (0, 0, 0): nothing was written there
  for (int box = 0; box < numBoxes; box++) {
    if (!(ranges[box] > rangeMinReturn)) continue;
    const Point& c = cpuPoints[box];
    const Point& g = gpuPoints[box];
    if (c.x == 0.0f && c.y == 0.0f && c.z == 0.0f) cpuAtZero++;
    if (g.x == 0.0f && g.y == 0.0f && g.z == 0.0f) gpuAtZero++;
    if (c.x == g.x && c.y == g.y && c.z == g.z) continue;
    differ++;
    largestGap = fmaxf(largestGap, fabsf(c.x - g.x));
    largestGap = fmaxf(largestGap, fabsf(c.y - g.y));
    largestGap = fmaxf(largestGap, fabsf(c.z - g.z));
  }
  printf("GPU vs CPU dots: %d of %d differ\n", differ, numReturns);
  printf("  largest difference in x, y or z: %f m\n", largestGap);
  printf("  dots left at (0, 0, 0): CPU %d, GPU %d\n", cpuAtZero, gpuAtZero);
  if (numReturns == 0 || cpuAtZero == numReturns || gpuAtZero == numReturns)
    printf("NOT A PASS: one side made no dots, so there is nothing to compare yet\n");

  // ---- Show the first few dots, to compare by eye with numbers worked out by hand ----
  int shown = 0;
  for (int box = 0; box < numBoxes && shown < 5; box++) {
    if (!(ranges[box] > rangeMinReturn)) continue;
    shown++;
    int row = box / W, col = box % W;
    printf("row %d, column %d, range %.4f m, tilt %.6f, turn %.6f\n"
           "    CPU (%.4f, %.4f, %.4f)   GPU (%.4f, %.4f, %.4f)\n",
           row, col, ranges[box], tiltOfRow[row], turnOfColumn[col],
           cpuPoints[box].x, cpuPoints[box].y, cpuPoints[box].z,
           gpuPoints[box].x, gpuPoints[box].y, gpuPoints[box].z);
  }

  // ---- Save the GPU's dots as a scene file, boxes with no return left out ----
  // All white for now. The viewer takes the file's path: ./viewer <this file>
  std::vector<Point> pts;
  std::vector<Color> colors;
  pts.reserve(numReturns);
  colors.reserve(numReturns);
  for (int box = 0; box < numBoxes; box++) {
    if (!(ranges[box] > rangeMinReturn)) continue;
    pts.push_back(gpuPoints[box]);
    colors.push_back(Color{255, 255, 255});
  }
  char scenePath[64];
  snprintf(scenePath, sizeof scenePath, "frame%04d_laser%d_points.bin", frame, laser);
  if (!pts.empty() && writeScene(scenePath, pts, colors)) printf("saved %s (%d dots)\n", scenePath, (int)pts.size());
  else printf("could not save %s\n", scenePath);

  return 0;
}
