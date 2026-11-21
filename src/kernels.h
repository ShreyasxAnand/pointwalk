#ifndef KERNELS_H
#define KERNELS_H

// The two step 2 kernels, moved here UNCHANGED from project.cu so that
// project.cu and viewer.cu can both use them. Only a .cu file may include this.
//   colorKernel    first pass:  one thread per dot, writes glued notes
//   resolveKernel  second pass: one thread per box, turns notes into colours

#include <cmath>
#include "camera.h"    // Camera
#include "image.h"     // Color
#include "pack.h"      // Packed, packDepthIndex, unpackDepth, unpackIndex
#include "point.h"     // Point

__global__ void colorKernel(const Point* pts, int N, Camera cam, Packed* notes) {
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
  if (px < 0 || px >= cam.W || py < 0 || py >= cam.H) return;   // off the photo

  int idx = py * cam.W + px;
  Packed glued = packDepthIndex(zc, i);   // glue: distance on the left, dot number on the right
  atomicMin(&notes[idx], glued);          // write it on this box's note, only if it is smaller
}

// Second pass. One thread = one BOX (not one dot). Reads the box's note and paints the box.
// No atomicMin: each thread writes only its own box, so nobody fights over anything.
__global__ void resolveKernel(const Packed* notes, const Color* colors, int numBoxes, Color* image) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;   // which box is mine?
  if (k >= numBoxes) return;                        // extra thread, no box

  Packed note = notes[k];                           // my box's note
  if (unpackDepth(note) >= 1e29f) {                 // still "really far": no dot landed here
    image[k] = Color{0, 0, 0};                      // paint it black
    return;
  }

  unsigned int dot = unpackIndex(note);             // which dot won this box
  image[k] = colors[dot];                           // paint the box that dot's colour
}

#endif
