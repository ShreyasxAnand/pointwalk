// One Waymo range image -> a point cloud in the vehicle frame (pointcloud.h).
// Usage: waymo_to_points <converted folder> <frame> <laser> <output file>
// Laser 1 is the top laser. Normally started by run.py.
// Exits with 1 on any error, after printing it to stderr.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../generic/color.h"              // Color
#include "../generic/point.h"              // Point
#include "../generic/pointcloud.h"          // writePointCloud
#include "../generic/geometry.h"           // rangeMinReturn
#include "../generic/range_image.h"        // SensorModel, RangeFrame
#include "../generic/launch.cuh"           // rangeToVehicleGPU
#include "waymo_frame.h"                   // loadWaymoSensor, loadWaymoFrame

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <converted folder> <frame> <laser> <output file>\n", argv[0]);
    return 1;
  }
  const char* folder = argv[1];
  const int frame = atoi(argv[2]);
  const int laser = atoi(argv[3]);
  const char* outPath = argv[4];

  // ---- 1. Load the sensor and the frame (loaders print their own errors) ----
  SensorModel sensor;
  RangeFrame rangeFrame;
  if (!loadWaymoSensor(folder, laser, sensor)) return 1;
  if (!loadWaymoFrame(folder, frame, laser, rangeFrame)) return 1;

  // ---- 2. Range image -> vehicle-frame dots, on the GPU ----
  std::vector<Point> boxPoints;
  if (!rangeToVehicleGPU(sensor, rangeFrame, boxPoints)) return 1;

  // ---- 3. Drop empty boxes, write ----
  const int numBoxes = rangeFrame.H * rangeFrame.W;
  std::vector<Point> pts;
  std::vector<Color> colors;
  for (int box = 0; box < numBoxes; box++) {
    if (!(rangeFrame.ranges[box] > rangeMinReturn)) continue;
    pts.push_back(boxPoints[box]);
    colors.push_back(rangeFrame.colors[box]);
  }
  if (pts.empty() || !writePointCloud(outPath, pts, colors)) {
    fprintf(stderr, "could not save %s\n", outPath);
    return 1;
  }
  printf("frame %d, laser %d: %d of %d boxes have a return -> saved %s\n",
         frame, laser, (int)pts.size(), numBoxes, outPath);
  return 0;
}
