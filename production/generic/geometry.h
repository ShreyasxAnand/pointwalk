#ifndef POINTWALK_GENERIC_GEOMETRY_H
#define POINTWALK_GENERIC_GEOMETRY_H

#include <cmath>          // sinf, cosf
#include "cpu_and_gpu.h"  // HD

// Math shared by the CPU and the GPU: every function is HD.
// Axes in every frame: +x right, +y down, +z forward. Distances in metres.
// Angles in radians. Tilt: 0 level, positive down. Turn: 0 straight ahead,
// positive right. Grids are stored row by row: box (row, col) is at row * W + col.

// ===== Step 7: unprojection, range image -> vehicle frame =====
// Tilts and turns come as lists, one per row and column, because a real
// laser's rows are not evenly spaced.

// A box whose range is at or below this has no return.
const float rangeMinReturn = 0.0f;

// The mount: where the sensor sits on the vehicle. Sensor -> vehicle is R * dot + t.
struct SensorToVehicle {
  float R[3][3];   // rotation
  float t[3];      // translation (metres)
};

// Direction and distance -> the dot as seen from the sensor (unprojection).
// In:  tilt and turn (radians), distance (metres).
// Out: the dot's x, y, z in the sensor frame (metres).
// Example: 30 deg down, 90 deg right, distance 2 -> (1.732, 1, 0).
HD inline void sphericalToSensor(float tilt, float turn, float distance,
                                 float& xs, float& ys, float& zs)
{
  xs = distance * sinf(turn) * cosf(tilt);
  ys = distance * sinf(tilt);
  zs = distance * cosf(turn) * cosf(tilt);
}

// A dot seen from the sensor -> the same dot on the vehicle.
// In:  the dot in the sensor frame (metres), the mount.
// Out: the dot in the vehicle frame (metres).
HD inline void sensorToVehicle(float xs, float ys, float zs, const SensorToVehicle& mount,
                               float& xv, float& yv, float& zv)
{
  xv = mount.R[0][0] * xs + mount.R[0][1] * ys + mount.R[0][2] * zs + mount.t[0];
  yv = mount.R[1][0] * xs + mount.R[1][1] * ys + mount.R[1][2] * zs + mount.t[1];
  zv = mount.R[2][0] * xs + mount.R[2][1] * ys + mount.R[2][2] * zs + mount.t[2];
}

#endif
