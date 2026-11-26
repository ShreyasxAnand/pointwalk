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

#include <cmath>       // atan2
#include <stdexcept>   // std::invalid_argument
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
  tiltOfRow.resize(H);
  if (cal.beamValues.size() != H) throw std::invalid_argument("beamValues size mismatch");
  for (int row = 0; row < H; row++) {
    tiltOfRow[row] = -static_cast<float>(cal.beamValues[H - 1 -row]); // flip tilt directions, waymo lists beams lowest first with +up, rangepoints.h lists highest first with +down, so reverse order and negative
  }
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
  turnOfColumn.resize(W);
  const double pi = 3.14159265358979323846;
  double az_correction = atan2(cal.extrinsic[4], cal.extrinsic[0]);
  for (int col = 0; col < W; col++) {
    turnOfColumn[col] = static_cast<float>(-(pi - (col + 0.5) * 2.0 * pi / W) + az_correction);
  }
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
  // Waymo's matrix, 16 numbers read row by row:
  //    [0]  [1]  [2]  | [3]     rows and columns are Waymo's x, y, z
  //    [4]  [5]  [6]  | [7]     the last column is the shift
  //    [8]  [9]  [10] | [11]
  //
  // The axes:   your x (right)   = minus Waymo's y (left)
  //             your y (down)    = minus Waymo's z (up)
  //             your z (forward) =       Waymo's x (forward)
  const double* E = cal.extrinsic;

  // The shift: Waymo's (x, y, z) shift is E[3], E[7], E[11].
  place.t[0] = static_cast<float>(-E[7]);    // right   = minus left
  place.t[1] = static_cast<float>(-E[11]);   // down    = minus up
  place.t[2] = static_cast<float>( E[3]);    // forward = forward

  // The rotation. A minus for every x or y of yours that is involved:
  // none or two of them -> plus, exactly one of them -> minus.
  place.R[0][0] = static_cast<float>( E[5]);    // your x from your x:  Waymo y from y
  place.R[0][1] = static_cast<float>( E[6]);    // your x from your y:  Waymo y from z
  place.R[0][2] = static_cast<float>(-E[4]);    // your x from your z:  Waymo y from x

  place.R[1][0] = static_cast<float>( E[9]);    // your y from your x:  Waymo z from y
  place.R[1][1] = static_cast<float>( E[10]);   // your y from your y:  Waymo z from z
  place.R[1][2] = static_cast<float>(-E[8]);    // your y from your z:  Waymo z from x

  place.R[2][0] = static_cast<float>(-E[1]);    // your z from your x:  Waymo x from y
  place.R[2][1] = static_cast<float>(-E[2]);    // your z from your y:  Waymo x from z
  place.R[2][2] = static_cast<float>( E[0]);    // your z from your z:  Waymo x from x
}

#endif
