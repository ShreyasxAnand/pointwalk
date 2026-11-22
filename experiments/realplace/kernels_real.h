#ifndef KERNELS_REAL_H
#define KERNELS_REAL_H

// Tonight's experiment: a first pass that paints each dot as a small square whose
// size depends on distance, so real lidar looks solid up close.
// It is colorKernel (src/kernels.h) with one change: instead of writing ONE note,
// it writes the same glued number onto a square of notes around the dot.
// Only a .cu file may include this.

#include <cmath>
#include "camera.h"    // Camera
#include "pack.h"      // Packed, packDepthIndex
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

#endif
