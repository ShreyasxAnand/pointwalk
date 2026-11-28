#ifndef POINTWALK_ADAPTERS_WAYMO_FRAME_H
#define POINTWALK_ADAPTERS_WAYMO_FRAME_H

// Converted Waymo files -> SensorModel and RangeFrame (generic/range_image.h).
// The only way into Waymo data: nothing outside adapters/ and waymo/ reads it.
// Both loaders print their own errors and return false on any problem.

#include <cmath>      // sqrtf, fminf, fmaxf
#include <cstdio>     // snprintf, fprintf
#include <vector>
#include "../waymo/waymofile.h"          // readWaymoRange, readWaymoCalibration
#include "../generic/range_image.h"      // SensorModel, RangeFrame
#include "../generic/color.h"            // Color
#include "waymo_adapter.h"               // waymoTiltOfRow, waymoTurnOfColumn, waymoSensorToVehicle

// Grey = waymoGreyDarkest + waymoGreySpan * sqrt(intensity, clamped to 0..1).
// The root spreads out the many weak returns (~0.1).
const float waymoGreyDarkest = 50.0f;    // the weakest return; keeps every dot visible
const float waymoGreySpan = 205.0f;      // darkest + span = 255, the strongest return

// One box's grey from its intensity.
// In:  intensity (Waymo channel 1, unitless, about 0 to 1).
// Out: a grey Color.
inline Color waymoGreyOfIntensity(float intensity) {
  const float s = sqrtf(fminf(fmaxf(intensity, 0.0f), 1.0f));
  const unsigned char grey = static_cast<unsigned char>(waymoGreyDarkest + waymoGreySpan * s);
  return Color{grey, grey, grey};
}

// The path of one converted range image: <folder>/range/frameFFFF_laserL.bin.
inline void waymoRangePath(char* path, size_t size, const char* folder, int frame, int laser) {
  snprintf(path, size, "%s/range/frame%04d_laser%d.bin", folder, frame, laser);
}

// One laser of a drive -> its SensorModel.
// In:  the converted folder, the laser (1 is the top laser).
// Out: sensor; false, with a message, on any problem.
// H and W come from frame 0's range image: convert.py always writes frame 0,
// and a laser's image size does not change during a drive.
inline bool loadWaymoSensor(const char* folder, int laser, SensorModel& sensor) {
  char calibrationPath[1024];
  snprintf(calibrationPath, sizeof calibrationPath, "%s/calibration.bin", folder);
  std::vector<WaymoLaserCalibration> calibrations;
  if (!readWaymoCalibration(calibrationPath, calibrations)) return false;

  const WaymoLaserCalibration* cal = nullptr;   // looked up by laser name, not file position
  for (const auto& c : calibrations)
    if (c.laser == laser) cal = &c;
  if (!cal) {
    fprintf(stderr, "no calibration for laser %d in %s\n", laser, calibrationPath);
    return false;
  }

  char rangePath[1024];
  waymoRangePath(rangePath, sizeof rangePath, folder, 0, laser);
  WaymoRangeImage image;
  if (!readWaymoRange(rangePath, image)) return false;
  sensor.H = image.H;
  sensor.W = image.W;

  if (!waymoTiltOfRow(*cal, sensor.H, sensor.tiltOfRow)) return false;
  waymoTurnOfColumn(*cal, sensor.W, sensor.turnOfColumn);
  waymoSensorToVehicle(*cal, sensor.mount);
  return true;
}

// One laser at one frame -> its RangeFrame.
// In:  the converted folder, the frame number (from 0), the laser.
// Out: frame: channel 0 as ranges (metres, -1 for no return), a grey per box
//      from channel 1 (white if there is no channel 1); false, with a message,
//      on any problem.
inline bool loadWaymoFrame(const char* folder, int frameNumber, int laser, RangeFrame& frame) {
  char rangePath[1024];
  waymoRangePath(rangePath, sizeof rangePath, folder, frameNumber, laser);
  WaymoRangeImage image;
  if (!readWaymoRange(rangePath, image)) return false;

  frame.H = image.H;
  frame.W = image.W;
  const size_t numBoxes = static_cast<size_t>(image.H) * image.W;
  frame.ranges.resize(numBoxes);
  frame.colors.resize(numBoxes);
  for (size_t box = 0; box < numBoxes; box++) {
    frame.ranges[box] = image.values[box * image.C];
    frame.colors[box] = image.C > 1 ? waymoGreyOfIntensity(image.values[box * image.C + 1])
                                    : Color{255, 255, 255};
  }
  return true;
}

#endif
