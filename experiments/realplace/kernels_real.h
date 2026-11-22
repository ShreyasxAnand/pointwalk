#ifndef KERNELS_REAL_H
#define KERNELS_REAL_H

// Tonight's experiment: two kernels for real lidar. Only a .cu file may include this.
//   splatKernel       first pass:  each dot is a small square sized by distance
//   resolveSkyKernel  second pass: empty boxes become sky
//
// splatKernel is colorKernel (src/kernels.h) with one change: instead of writing ONE
// note, it writes the same glued number onto a square of notes around the dot.

#include <cmath>
#include "camera.h"    // Camera
#include "image.h"     // Color
#include "pack.h"      // Packed, packDepthIndex, unpackDepth, unpackIndex
#include "point.h"     // Point

// dotSize: how wide one dot is in the real world, in metres (0.3 = 30 cm).
// On screen that is  fx * dotSize / distance  boxes wide: big when near, 1 box when far.
__global__ void splatKernel(const Point* pts, int N, Camera cam, Packed* notes, float dotSize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // which dot is mine?
  if (i >= N) return;                               // extra thread, no dot

  Point p = pts[i];
  float xc = cam.R[0][0]*p.x + cam.R[0][1]*p.y + cam.R[0][2]*p.z + cam.t[0];
  float yc = cam.R[1][0]*p.x + cam.R[1][1]*p.y + cam.R[1][2]*p.z + cam.t[1];
  float zc = cam.R[2][0]*p.x + cam.R[2][1]*p.y + cam.R[2][2]*p.z + cam.t[2];
  if (zc < 0.1f) return;                            // behind camera

  float u = cam.fx * (xc / zc) + cam.cx;
  float v = cam.fy * (yc / zc) + cam.cy;
  int px = (int)floorf(u);
  int py = (int)floorf(v);

  int size = (int)ceilf(cam.fx * dotSize / zc);     // how many boxes wide this dot is from here
  if (size < 1) size = 1;
  if (size > 8) size = 8;                           // cap it, so one very close dot cannot cost 1000 writes
  if (px + size < 0 || px - size >= cam.W || py + size < 0 || py - size >= cam.H) return;   // fully off the photo

  Packed glued = packDepthIndex(zc, i);             // same glued number for every box of the square
  for (int dy = 0; dy < size; dy++) {
    for (int dx = 0; dx < size; dx++) {
      int ax = px + dx - size / 2;                  // the square is centred on the dot
      int ay = py + dy - size / 2;
      if (ax < 0 || ax >= cam.W || ay < 0 || ay >= cam.H) continue;   // this box is off the photo
      atomicMin(&notes[ay * cam.W + ax], glued);    // closer? overwrite the note
    }
  }
}

// Second pass with a sky. It is resolveKernel (src/kernels.h) with one change: a box
// whose note is empty is painted as sky, not black. Which sky colour depends on
// whether that box looks above or below the horizon.
__global__ void resolveSkyKernel(const Packed* notes, const Color* colors, Camera cam, Color* image) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;   // which box is mine?
  if (k >= cam.W * cam.H) return;                   // extra thread, no box

  Packed note = notes[k];                           // my box's note
  if (unpackDepth(note) < 1e29f) {                  // a dot landed here: paint its colour
    image[k] = colors[unpackIndex(note)];
    return;
  }

  // Empty note. Work out which way this box looks: the pinhole formula run backwards
  // gives the direction in the camera's view, and R turns that into a world direction.
  float a = ((k % cam.W) + 0.5f - cam.cx) / cam.fx;    // how far right of the centre, per metre forward
  float b = ((k / cam.W) + 0.5f - cam.cy) / cam.fy;    // how far below the centre, per metre forward
  float worldDown = cam.R[0][1]*a + cam.R[1][1]*b + cam.R[2][1];    // the direction's world y (y points down)
  float up = -worldDown / sqrtf(a*a + b*b + 1.0f);     // +1 straight up, 0 at the horizon, -1 straight down

  // Above the horizon: pale at the horizon, blue overhead. Below it: pale, fading to a dull ground haze.
  float blend = (up > 0.0f) ? sqrtf(up) : fminf(1.0f, -up * 4.0f);
  float farR = (up > 0.0f) ?  70.0f : 120.0f;
  float farG = (up > 0.0f) ? 130.0f : 115.0f;
  float farB = (up > 0.0f) ? 210.0f : 105.0f;
  Color sky;
  sky.r = (unsigned char)(200.0f + (farR - 200.0f) * blend);
  sky.g = (unsigned char)(220.0f + (farG - 220.0f) * blend);
  sky.b = (unsigned char)(240.0f + (farB - 240.0f) * blend);
  image[k] = sky;
}

#endif
