#ifndef RANGEPOINTS_H
#define RANGEPOINTS_H

#include <cmath>   // sinf, cosf
#include <vector>
#include "point.h"   // Point

// Step 7: a range image -> dots. The reverse of the 360 picture in pano.h:
// there a dot became a box and a distance, here a box and a distance become a
// dot. Plain C++, no CUDA in here, so it can be tested on this machine with g++.
//
// NOTHING IN THIS FILE KNOWS ABOUT WAYMO. It gets plain arrays, and
// io/waymo/waymo_adapter.h is what makes them from Waymo's files.
//
// SCAFFOLD: every function body below is empty. They are yours to write.

// HD in front of a function tells the CUDA compiler "build this for the CPU and
// the GPU". Under plain g++, HD means nothing. (Same macro as in pano.h.)
#ifndef HD
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

// THE INPUT: a range image H boxes tall and W boxes wide
//
//   ranges         H * W distances in metres, row after row:
//                    the box at (row, col) is ranges[row * W + col]
//   tiltOfRow      H angles in radians, one per row: how far that row's beam
//                    is tilted. 0 is level, positive is DOWN (y points down).
//   turnOfColumn   W angles in radians, one per column: how far that column
//                    is turned. 0 is straight ahead, positive is to the RIGHT.
//
// The angles come from two lists, not from a formula, because a real laser's
// rows are not evenly spaced. Same conventions as panoProject, so a dot made
// here and put through panoProject lands back on its own tilt and turn.

// A box holds a return only when its range is greater than this. The CPU
// version, the GPU kernel and main() must all use this name, never their own
// number, so all three apply the identical rule. (Waymo marks an empty box
// with -1.)
const float rangeMinReturn = 0.0f;

// Where the sensor sits in the scene: a turn and a shift, like R and t in
// Camera, but going the other way (Camera is world -> sensor).
struct SensorToScene {
  float R[3][3];   // rotation: sensor -> scene
  float t[3];      // shift:    sensor -> scene, in metres
};

// FUNCTION A. panoUnproject: where is the dot, seen from the sensor?
// This is panoProject (pano.h) run backwards.
//
// INPUTS
//   tilt       radians. 0 level, positive down.
//   turn       radians. 0 straight ahead, positive to the right.
//   distance   the straight-line distance to the dot, in metres
//
// OUTPUTS (passed by reference, so you fill in the caller's variables)
//   x, y, z    the dot, in steps from the sensor: x right, y DOWN, z ahead
// This is diferent from waymo input data which uses x forward, y left, z up.
//
// WORKED EXAMPLE (put yours here, then use it in the test)
//   tilt 30 degrees down, turn 90 degrees right, distance 2  ->  ( ?, ?, ? )
HD inline void panoUnproject(float tilt, float turn, float distance,
                             float& x, float& y, float& z)
{
  z = distance * cosf(turn) * cosf(tilt);
  x = distance * sinf(turn) * cosf(tilt);
  y = distance * sinf(tilt);
}

// FUNCTION B. sensorToScene: the same dot, measured in the scene instead of
// from the sensor. The reverse direction of worldToSensor (camera.h).
//
// INPUTS
//   xs, ys, zs   the dot in steps from the sensor (what function A gives)
//   place        where the sensor sits in the scene
//
// OUTPUTS
//   x, y, z      the dot in scene coordinates, in metres
HD inline void sensorToScene(float xs, float ys, float zs, const SensorToScene& place,
                             float& x, float& y, float& z)
{
  x = place.R[0][0] * xs + place.R[0][1] * ys + place.R[0][2] * zs + place.t[0];
  y = place.R[1][0] * xs + place.R[1][1] * ys + place.R[1][2] * zs + place.t[1];
  z = place.R[2][0] * xs + place.R[2][1] * ys + place.R[2][2] * zs + place.t[2];
}

// rangeToPointsCPU: turn a whole range image into dots, on the CPU.
// This is the slow, obviously-right version the GPU kernel gets compared with.
//
// INPUTS
//   ranges, H, W, tiltOfRow, turnOfColumn   the range image, see the top of this file
//   place                                   where the sensor sits in the scene
//
// OUTPUT
//   points     H * W dots, one per box, in the same order as ranges. The caller
//              has already made it H * W long and filled it with zeros.
//
// THE JOBS, in order, for each box
//   1. box number -> row and column
//   2. read the box's range. No return -> nothing to do for this box.
//   3. the row's tilt and the column's turn, from the two lists
//   4. function A -> the dot, seen from the sensor
//   5. function B -> the dot, in the scene
//   6. store it in points, at the box's own position
//
// DECIDE BEFORE YOU WRITE
//   a. What is in points[box] for a box with no return? main() never reads
//      those boxes, but your GPU kernel must leave the same thing there.
inline void rangeToPointsCPU(const std::vector<float>& ranges, int H, int W,
                             const std::vector<float>& tiltOfRow,
                             const std::vector<float>& turnOfColumn,
                             const SensorToScene& place, std::vector<Point>& points)
{
  for (int row = 0; row < H; row++) {
    for (int col = 0; col < W; col++) {
      float range = ranges[row * W + col];
      if (range <= rangeMinReturn) continue;
      float tilt = tiltOfRow[row];
      float turn = turnOfColumn[col];
      float xs, ys, zs;
      panoUnproject(tilt, turn, range, xs, ys, zs);

      float x, y, z;
      sensorToScene(xs, ys, zs, place, x, y, z);

      points[row * W + col].x = x;
      points[row * W + col].y = y;
      points[row * W + col].z = z;
    }
  }
}

#endif
