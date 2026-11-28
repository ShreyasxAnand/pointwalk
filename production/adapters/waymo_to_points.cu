// One Waymo range image -> a point cloud in the vehicle frame (pointcloud.h).
// Usage: waymo_to_points <converted folder> <frame> <laser> <output file>
// Laser 1 is the top laser. Normally started by run.py.
// Exits with 1 on any error, after printing it to stderr.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../generic/color.h"              // Color
#include "../generic/point.h"              // Point
#include "../generic/pointcloud.h"          // writePointCloud
#include "../generic/kernels.cuh"          // rangeMinReturn
#include "../generic/launch.cuh"           // rangeToVehicleGPU
#include "../waymo/waymofile.h"            // readWaymoRange, readWaymoCalibration
#include "waymo_adapter.h"                 // waymoTiltOfRow, waymoTurnOfColumn, waymoSensorToVehicle

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <converted folder> <frame> <laser> <output file>\n", argv[0]);
    return 1;
  }
  const char* folder = argv[1];
  const int frame = atoi(argv[2]);
  const int laser = atoi(argv[3]);
  const char* outPath = argv[4];

  // ---- 1. Read the range image and calibration ----
  char rangePath[1024], calibrationPath[1024];
  snprintf(rangePath, sizeof rangePath, "%s/range/frame%04d_laser%d.bin", folder, frame, laser);
  snprintf(calibrationPath, sizeof calibrationPath, "%s/calibration.bin", folder);

  WaymoRangeImage image;
  std::vector<WaymoLaserCalibration> calibrations;
  if (!readWaymoRange(rangePath, image)) return 1;                    // readers print their own errors
  if (!readWaymoCalibration(calibrationPath, calibrations)) return 1;

  const WaymoLaserCalibration* cal = nullptr;   // looked up by laser name, not file position
  for (const auto& c : calibrations)
    if (c.laser == laser) cal = &c;
  if (!cal) {
    fprintf(stderr, "no calibration for laser %d in %s\n", laser, calibrationPath);
    return 1;
  }

  const int H = image.H, W = image.W;
  const int numBoxes = H * W;
  std::vector<float> ranges(numBoxes);   // channel 0 is the range (metres)
  for (int box = 0; box < numBoxes; box++) ranges[box] = image.values[(size_t)box * image.C];

  // ---- 2. Calibration -> tilts, turns, mount ----
  std::vector<float> tiltOfRow, turnOfColumn;
  SensorToVehicle mount;
  if (!waymoTiltOfRow(*cal, H, tiltOfRow)) return 1;   // prints its own error
  waymoTurnOfColumn(*cal, W, turnOfColumn);
  waymoSensorToVehicle(*cal, mount);

  // ---- 3. Range image -> vehicle-frame dots, on the GPU ----
  std::vector<Point> boxPoints = rangeToVehicleGPU(ranges, H, W, tiltOfRow, turnOfColumn, mount);

  // ---- 4. Drop empty boxes, grey from intensity, write ----
  // Grey = 50 + 205 * sqrt(intensity, clamped to 0..1), intensity is channel 1.
  // The root spreads out the many weak returns (~0.1); 50 keeps every dot visible.
  std::vector<Point> pts;
  std::vector<Color> colors;
  for (int box = 0; box < numBoxes; box++) {
    if (!(ranges[box] > rangeMinReturn)) continue;
    pts.push_back(boxPoints[box]);
    unsigned char grey = 255;
    if (image.C > 1) {
      float intensity = image.values[(size_t)box * image.C + 1];
      float s = sqrtf(fminf(fmaxf(intensity, 0.0f), 1.0f));
      grey = (unsigned char)(50.0f + 205.0f * s);
    }
    colors.push_back(Color{grey, grey, grey});
  }
  if (pts.empty() || !writePointCloud(outPath, pts, colors)) {
    fprintf(stderr, "could not save %s\n", outPath);
    return 1;
  }
  printf("frame %d, laser %d: %d of %d boxes have a return -> saved %s\n",
         frame, laser, (int)pts.size(), numBoxes, outPath);
  return 0;
}
