// One Waymo range image -> a point cloud in the vehicle frame (pointcloud.h).
// Usage: waymo_to_points <converted folder> <frame> <laser> <output file>
// Laser 1 is the top laser. Normally started by run.py.
// Exits with 1 on any error, after printing it to stderr.

#include <cerrno>     // errno, ERANGE
#include <climits>    // INT_MIN, INT_MAX
#include <cstdio>
#include <cstdlib>    // strtol
#include <string>
#include <vector>
#include "generic/color.h"              // Color
#include "generic/point.h"              // Point
#include "generic/pointcloud.h"         // writePointCloud
#include "generic/geometry.h"           // rangeMinReturn
#include "generic/range_image.h"        // SensorModel, RangeFrame
#include "generic/launch.cuh"           // rangeToVehicleGPU
#include "adapters/waymo_frame.h"       // loadWaymoSensor, loadWaymoFrame

const int firstLaser = 1;   // Waymo's laser names: 1 top, 2 front, 3 side left,
const int lastLaser = 5;    // 4 side right, 5 rear

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <converted folder> <frame> <laser> <output file>\n", argv[0]);
    return 1;
  }

  // A whole base-10 number with nothing after it, that fits in an int.
  auto parseInt = [](const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
    value = static_cast<int>(v);
    return true;
  };

  const std::string folder = argv[1];
  const std::string outPath = argv[4];
  int frame = 0, laser = 0;
  if (!parseInt(argv[2], frame)) {
    fprintf(stderr, "frame must be a whole number, got \"%s\"\n", argv[2]);
    return 1;
  }
  if (frame < 0) {
    fprintf(stderr, "frame must be 0 or more, got %d\n", frame);
    return 1;
  }
  if (!parseInt(argv[3], laser)) {
    fprintf(stderr, "laser must be a whole number, got \"%s\"\n", argv[3]);
    return 1;
  }
  if (laser < firstLaser || laser > lastLaser) {
    fprintf(stderr, "laser must be %d to %d, got %d\n", firstLaser, lastLaser, laser);
    return 1;
  }

  // ---- 1. Load the sensor and the frame (loaders print their own errors) ----
  SensorModel sensor;
  RangeFrame rangeFrame;
  if (!loadWaymoSensor(folder.c_str(), laser, sensor)) return 1;
  if (!loadWaymoFrame(folder.c_str(), frame, laser, rangeFrame)) return 1;

  // ---- 2. Range image -> vehicle-frame dots, on the GPU ----
  std::vector<Point> boxPoints;
  if (!rangeToVehicleGPU(sensor, rangeFrame, boxPoints)) return 1;

  // ---- 3. Drop empty boxes, write (the writer prints its own errors) ----
  const size_t numBoxes = boxPoints.size();
  std::vector<Point> pts;
  std::vector<Color> colors;
  pts.reserve(numBoxes);
  colors.reserve(numBoxes);
  for (size_t box = 0; box < numBoxes; box++) {
    if (!(rangeFrame.ranges[box] > rangeMinReturn)) continue;
    pts.push_back(boxPoints[box]);
    colors.push_back(rangeFrame.colors[box]);
  }
  if (!writePointCloud(outPath.c_str(), pts, colors)) return 1;
  printf("frame %d, laser %d: %d of %d boxes have a return -> saved %s\n",
         frame, laser, static_cast<int>(pts.size()), static_cast<int>(numBoxes), outPath.c_str());
  return 0;
}
