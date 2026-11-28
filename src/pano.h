#ifndef PANO_H
#define PANO_H

#include <cmath>   // atan2f, sqrtf, floorf
#include <vector>
#include "camera.h"   // Camera: only R and t are used for the 360 picture
#include "point.h"    // Point

// Step 4: the wrap-around (360 degree) picture. Which box does a dot land in,
// and how far away is it? Plain C++, no CUDA in here, so it can be tested on
// this machine with g++.

// HD in front of a function tells the CUDA compiler "build this for the CPU and
// the GPU". Under plain g++, HD means nothing. (Same macro as in pack.h.)
#ifndef HD
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

// THE LAYOUT of a picture W boxes wide and H boxes tall
//
//   Columns say how far you have turned:
//     left edge  (column 0)    directly behind you, reached by turning left
//     middle     (column W/2)  straight ahead
//     right edge (column W)    directly behind you, reached by turning right
//                              (the same direction as column 0: the picture wraps)
//
//   Rows say how far you have tilted:
//     top row    (row 0)       straight up
//     middle     (row H/2)     level
//     bottom     (row H)       straight down
//
//   The number stored in a box is the straight-line distance to the dot.

// The nearest a dot may be, in metres. Anything closer is left out of the
// picture, for two reasons:
//   - a dot exactly on the sensor has no direction (atan2f(0, 0) is a made-up 0)
//   - at distance 0 it would win its box and hide the real nearest dot
// It is the same idea as the flat camera's "if (zc < 0.1f)". The CPU version
// below and the GPU kernel in pano.cu must both use this name, never their own
// number, so the two pictures apply the identical rule.
const float panoMinDistance = 0.1f;

// sensorToPano: where does a dot land in the wrap-around picture?
//
// INPUTS
//   x, y, z    the dot, in steps from the sensor: x right, y DOWN, z ahead, in metres.
//              (These are the sensor's own directions, the same ones xc, yc, zc
//              have in the flat photo kernel.)
//   W, H       the picture's size in boxes
//
// OUTPUTS (passed by reference, so you fill in the caller's variables)
//   u          the column, as a decimal number (217.87 means box 217)
//   v          the row, as a decimal number
//   distance   the straight-line distance to the dot, in metres
HD inline void sensorToPano(float x, float y, float z, int W, int H,
                           float& u, float& v, float& distance)
{
  const float pi = 3.14159265f;

  u = (atan2f(x, z) + pi) * W / (2 * pi);
  distance = sqrtf(x * x + y * y + z * z);
  float horizontalDistance = sqrtf(x * x + z * z);
  float tilt = atan2f(y, horizontalDistance);   // -pi/2 straight up, 0 level, +pi/2 straight down
  v = (tilt + pi / 2) * H / pi;
}

// panoDepthCPU: draw the whole scene as a 360 depth picture, on the CPU.
// This is the slow, obviously-right version the GPU kernel gets compared with.
//
// INPUTS
//   pts        the dots, in world coordinates
//   cam        where the sensor stands and which way it faces. Only cam.R and
//              cam.t are used (world -> sensor, the same three lines as in
//              depthKernel). The lens numbers fx, fy, cx, cy do not apply here.
//   W, H       the picture's size in boxes
//
// OUTPUT
//   depth      W * H numbers, one per box, row after row. The caller has
//              already filled it with 1e30 ("really far"). A box that no dot
//              lands in must still say 1e30 afterwards.
//
// THE JOBS, in order, for each dot
//   1. world -> sensor (R and t)
//   2. sensorToPano -> column, row, distance
//   3. decimal column and row -> whole box numbers
//   4. box numbers -> position in the long list
//   5. keep the smaller distance
//
// DECIDE BEFORE YOU WRITE (these are the places where it can go wrong)
//   a. A dot directly behind can give u = W exactly. There is no column W.
//      Column W and column 0 are the same direction.
//   b. A dot straight down gives v = H exactly. There is no row H.
//   c. A dot sitting exactly on the sensor has distance 0 and no direction.
//      DECIDED: skip every dot nearer than panoMinDistance.
inline void panoDepthCPU(const std::vector<Point>& pts, const Camera& cam,
                         int W, int H, std::vector<float>& depth)
{
  for (const auto& pt : pts) {
    float x, y, z;
    float u, v, distance;
    worldToSensor(pt.x, pt.y, pt.z, cam, x, y, z);
    sensorToPano(x, y, z, W, H, u, v, distance);
    if (distance < panoMinDistance) continue;   // too close to be real: skip
    int col = static_cast<int>(u);
    int row = static_cast<int>(v);
    if (col == W) col = 0;
    if (row == H) row = H - 1;
    int index = row * W + col;
    if (distance < depth[index]) {
      depth[index] = distance;
    }
  }
}

#endif
