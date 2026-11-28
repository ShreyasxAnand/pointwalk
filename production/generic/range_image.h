#ifndef POINTWALK_GENERIC_RANGE_IMAGE_H
#define POINTWALK_GENERIC_RANGE_IMAGE_H

#include <vector>
#include "color.h"      // Color
#include "geometry.h"   // SensorToVehicle; axes, angle signs and grid order

// A range image split in two: what is fixed for a drive, and what each frame
// brings. Conventions (axes, tilt and turn signs, row-by-row grids) are the
// ones at the top of geometry.h.

// ===== Step 7: unprojection, range image -> vehicle frame =====

// One laser: fixed for a drive.
struct SensorModel {
  int H, W;                          // rows, columns of every range image
  std::vector<float> tiltOfRow;      // H tilts (radians, 0 level, positive down)
  std::vector<float> turnOfColumn;   // W turns (radians, 0 straight ahead, positive right)
  SensorToVehicle mount;             // sensor -> vehicle: R, and t in metres
};

// One laser's picture at one moment: one per frame.
struct RangeFrame {
  int H, W;                     // rows, columns
  std::vector<float> ranges;    // H * W distances (metres), box (row, col) at row * W + col;
                                // at or below rangeMinReturn: no return
  std::vector<Color> colors;    // H * W colours, one per box, same order
};

#endif
