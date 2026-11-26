#ifndef WAYMO_ADAPTER_H
#define WAYMO_ADAPTER_H

// The Waymo adapter: turns what waymofile.h reads (raw, exactly as Waymo
// stores it) into the plain inputs src/rangepoints.h wants. Everything that is
// particular to Waymo lives here, so the kernel never has to know about it.
// Plain C++, no CUDA in here, so it can be tested on this machine with g++.
//
// SCAFFOLD: every function body below is empty. They are yours to write.
// The raw facts are in waymo/DATA_FORMAT.md. The four questions it leaves open
// ("Not in this file, on purpose") are the ones these functions answer.

#include <vector>
#include "waymofile.h"                // WaymoLaserCalibration
#include "../../src/rangepoints.h"    // SensorToScene

// waymoTiltOfRow: the tilt of every image row, for function A.
//
// INPUTS
//   cal          one laser's calibration, raw
//   H            the number of rows in that laser's range image
//
// OUTPUT
//   tiltOfRow    make it H long. tiltOfRow[row] in radians, 0 level,
//                positive DOWN (rangepoints.h's convention, not Waymo's).
//
// DECIDE BEFORE YOU WRITE
//   a. Which image row goes with which entry of cal.beamValues?
//   b. Which way does Waymo count a tilt as positive, and which way do you?
//   c. cal.beamValues is EMPTY for lasers 2 to 5. What then? (Laser 1 first:
//      it is fine to refuse the others for now, as long as you say so.)
inline void waymoTiltOfRow(const WaymoLaserCalibration& cal, int H,
                           std::vector<float>& tiltOfRow)
{
  // yours
}

// waymoTurnOfColumn: the turn of every image column, for function A.
//
// INPUTS
//   cal            one laser's calibration, raw
//   W              the number of columns in that laser's range image
//
// OUTPUT
//   turnOfColumn   make it W long. turnOfColumn[col] in radians, 0 straight
//                  ahead, positive to the RIGHT (rangepoints.h's convention).
//
// DECIDE BEFORE YOU WRITE
//   a. Which column looks straight ahead, and which way do the columns turn
//      as the column number goes up?
//   b. Is a column's direction its left edge or its middle?
//   c. "Straight ahead" of what: the laser or the car? The laser is mounted
//      turned on the roof (see its extrinsic in DATA_FORMAT.md).
inline void waymoTurnOfColumn(const WaymoLaserCalibration& cal, int W,
                              std::vector<float>& turnOfColumn)
{
  // yours
}

// waymoSensorToVehicle: where the laser sits on the car, for function B.
//
// INPUT
//   cal      one laser's calibration, raw. cal.extrinsic is 16 doubles, the
//            matrix read row by row, in WAYMO's axes (x forward, y left, z up).
//
// OUTPUT
//   place    sensor -> vehicle, in YOUR axes (x right, y down, z forward),
//            on both sides: function A hands it a dot in your axes, and the
//            dot that comes out must be in your axes too.
//
// DECIDE BEFORE YOU WRITE
//   a. How does a direction in Waymo's axes read in your axes?
//   b. The matrix expects Waymo's axes going in and gives Waymo's axes coming
//      out. What has to happen on each side of it?
//   c. Do the arithmetic in double, and turn the result into float last.
inline void waymoSensorToVehicle(const WaymoLaserCalibration& cal, SensorToScene& place)
{
  // yours
}

#endif
