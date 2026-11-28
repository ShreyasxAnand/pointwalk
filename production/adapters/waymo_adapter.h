#ifndef POINTWALK_ADAPTERS_WAYMO_ADAPTER_H
#define POINTWALK_ADAPTERS_WAYMO_ADAPTER_H

// Waymo adapter: one laser's raw calibration -> tilts, turns and the mount for
// geometry.h. The only code that knows Waymo's conventions.
// Waymo axes: x forward, y left, z up. Raw file layout: waymo/DATA_FORMAT.md.

#include <cmath>       // atan2
#include <cstdio>      // fprintf
#include <vector>
#include "../waymo/waymofile.h"       // WaymoLaserCalibration
#include "../generic/geometry.h"     // SensorToVehicle

// The tilt of each row.
// In:  the laser's calibration, the number of rows.
// Out: one tilt per row (radians); false, with a message, if the table size is wrong.
// Waymo lists beams lowest first, up positive; row 0 is the top: reverse, negate.
inline bool waymoTiltOfRow(const WaymoLaserCalibration& cal, int H,
                           std::vector<float>& tiltOfRow)
{
  if (static_cast<int>(cal.beamValues.size()) != H) {
    fprintf(stderr, "laser %d: beam table has %d entries, the range image has %d rows "
                    "(lasers 2 to 5 have no beam table and are not supported yet)\n",
            cal.laser, static_cast<int>(cal.beamValues.size()), H);
    return false;
  }
  tiltOfRow.resize(H);
  for (int row = 0; row < H; row++) {
    tiltOfRow[row] = -static_cast<float>(cal.beamValues[H - 1 - row]);
  }
  return true;
}

// The turn of each column, measured from the sensor's own forward.
// In:  the laser's calibration, the number of columns.
// Out: one turn per column (radians, rising right; Waymo's azimuth formula, sign flipped).
// At column centres; middle = vehicle forward. Adds the mount's yaw; mount.R removes it.
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

// The mount: Waymo's sensor -> vehicle matrix, rewritten in Pointwalk axes.
// In:  the laser's calibration (extrinsic: 4x4, row by row, Waymo axes).
// Out: the mount, Pointwalk axes on both sides.
inline void waymoSensorToVehicle(const WaymoLaserCalibration& cal, SensorToVehicle& mount)
{
  // Waymo's matrix, row-major:
  //    [0]  [1]  [2]  | [3]     rows and columns: Waymo x, y, z
  //    [4]  [5]  [6]  | [7]     last column: translation
  //    [8]  [9]  [10] | [11]
  //
  // Axis mapping:  Pointwalk x (right)   = -Waymo y (left)
  //                Pointwalk y (down)    = -Waymo z (up)
  //                Pointwalk z (forward) = +Waymo x (forward)
  const double* E = cal.extrinsic;

  mount.t[0] = static_cast<float>(-E[7]);    // right   = -left
  mount.t[1] = static_cast<float>(-E[11]);   // down    = -up
  mount.t[2] = static_cast<float>( E[3]);    // forward = forward

  // Rotation. Each Pointwalk x or y index carries a minus sign: none or both
  // -> plus, exactly one -> minus.
  mount.R[0][0] = static_cast<float>( E[5]);    // x from x:  Waymo y from y
  mount.R[0][1] = static_cast<float>( E[6]);    // x from y:  Waymo y from z
  mount.R[0][2] = static_cast<float>(-E[4]);    // x from z:  Waymo y from x

  mount.R[1][0] = static_cast<float>( E[9]);    // y from x:  Waymo z from y
  mount.R[1][1] = static_cast<float>( E[10]);   // y from y:  Waymo z from z
  mount.R[1][2] = static_cast<float>(-E[8]);    // y from z:  Waymo z from x

  mount.R[2][0] = static_cast<float>(-E[1]);    // z from x:  Waymo x from y
  mount.R[2][1] = static_cast<float>(-E[2]);    // z from y:  Waymo x from z
  mount.R[2][2] = static_cast<float>( E[0]);    // z from z:  Waymo x from x
}

#endif
