#include <cstdio>
#include <vector>
#include <cmath>
#include "camera.h"    // Camera and setPose
#include "image.h"     // writePGM
#include "pano.h"      // panoProject and panoDepthCPU
#include "point.h"     // Point
#include "scenefile.h" // readScene: loads the dots project.cu saved

// Step 4, second half: the 360 depth picture on the GPU, checked against the
// CPU version in pano.h.
//
// Run order: project first (it writes scene.bin), then this program from the
// same folder. The launcher notebook does that.

// ---- Settings ----
const int PW = 1024;   // picture width in boxes: a full turn
const int PH = 512;    // picture height in boxes: straight up to straight down

// Where the sensor stands, in metres, and which way it faces, in radians.
// (0, 0, 15) is the middle of the floor: floor all around, the ball behind.
const float sensorX = 0.0f, sensorY = 0.0f, sensorZ = 15.0f;
const float sensorYaw = 0.0f, sensorPitch = 0.0f;

// Copied unchanged from project.cu.
__device__ void atomicMinFloat(float*address, float val) {
    atomicMin((int*)address, __float_as_int(val));
}

// panoDepthKernel: the GPU version of panoDepthCPU. One thread = one dot.
//
// Same five jobs as the CPU version (see pano.h), with two differences you
// already know from depthKernel:
//   - no loop over the dots: each thread works out which dot is its own
//   - "keep the smaller distance" must be atomicMinFloat, because many
//     threads can land in the same box at the same time
//
// It must make the same decisions as your CPU version for the three special
// cases, or the two pictures will not match:
//   u = W   -> column 0
//   v = H   -> row H - 1
//   nearer than panoMinDistance (the named number in pano.h) -> skip the dot.
//   A kernel skips with "return" where the CPU loop uses "continue".
__global__ void panoDepthKernel(const Point* pts, int N, Camera cam,
                                int W, int H, float* depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  Point pt = pts[idx];
  float x, y, z;
  float u, v, distance;
  worldToSensor(pt.x, pt.y, pt.z, cam, x, y, z);
  panoProject(x, y, z, W, H, u, v, distance);
  if (distance < panoMinDistance) return;   // too close to be real: skip
  int col = static_cast<int>(u);
  int row = static_cast<int>(v);
  if (col == W) col = 0;
  if (row == H) row = H - 1;
  int index = row * W + col;
  atomicMinFloat(&depth[index], distance);
}

int main() {
  // ---- Load the dots ----
  std::vector<Point> pts;
  std::vector<Color> colors;   // loaded with the dots, not used here
  if (!readScene("scene.bin", pts, colors)) {
    printf("could not read scene.bin: run project first, from the same folder\n");
    return 1;
  }
  const int N = (int)pts.size();
  printf("loaded %d dots\n", N);

  // ---- The sensor: only R and t matter here, the lens numbers are unused ----
  Camera cam = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}, 0, 0, 0, 0, PW, PH};
  setPose(cam, sensorX, sensorY, sensorZ, sensorYaw, sensorPitch);

  const int numBoxes = PW * PH;

  // ---- CPU picture ----
  std::vector<float> cpuDepth(numBoxes, 1e30f);        // every box starts "really far"
  panoDepthCPU(pts, cam, PW, PH, cpuDepth);

  // ---- GPU picture ----
  Point* d_pts;
  float* d_depth;
  cudaMalloc(&d_pts, N * sizeof(Point));
  cudaMalloc(&d_depth, numBoxes * sizeof(float));

  std::vector<float> gpuDepth(numBoxes, 1e30f);        // starting values: "really far"
  cudaMemcpy(d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_depth, gpuDepth.data(), numBoxes * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;            // enough blocks to cover all N dots
  panoDepthKernel<<<blocks, threads>>>(d_pts, N, cam, PW, PH, d_depth);
  cudaDeviceSynchronize();
  printf("kernel status: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(gpuDepth.data(), d_depth, numBoxes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_pts);
  cudaFree(d_depth);

  // ---- Compare the two pictures, box by box ----
  int cpuFilled = 0, gpuFilled = 0;   // boxes with a dot in them
  int differ = 0;                     // boxes where the two numbers are not identical
  int filledInOneOnly = 0;            // of those: one picture has a dot, the other is empty
  float largestGap = 0.0f;            // of the rest: the biggest difference, in metres
  for (int k = 0; k < numBoxes; k++) {
    bool inCpu = cpuDepth[k] < 1e29f;
    bool inGpu = gpuDepth[k] < 1e29f;
    if (inCpu) cpuFilled++;
    if (inGpu) gpuFilled++;
    if (cpuDepth[k] == gpuDepth[k]) continue;
    differ++;
    if (inCpu != inGpu) filledInOneOnly++;
    else largestGap = fmaxf(largestGap, fabsf(cpuDepth[k] - gpuDepth[k]));
  }
  printf("CPU picture: %d of %d boxes have a dot\n", cpuFilled, numBoxes);
  printf("GPU picture: %d of %d boxes have a dot\n", gpuFilled, numBoxes);
  printf("GPU vs CPU 360: %d of %d boxes differ\n", differ, numBoxes);
  printf("  of those, filled in only one picture: %d\n", filledInOneOnly);
  printf("  largest difference where both have a dot: %f m\n", largestGap);
  if (cpuFilled == 0 || gpuFilled == 0)
    printf("NOT A PASS: a picture is empty, so there is nothing to compare yet\n");

  // ---- Save both pictures as grey images (near = white, far = dark, empty = black) ----
  const float dMin = 1.0f, dMax = 30.0f;
  std::vector<unsigned char> gray(numBoxes);
  const std::vector<float>* pictures[2] = {&cpuDepth, &gpuDepth};
  const char* names[2] = {"pano_cpu.pgm", "pano_gpu.pgm"};
  for (int which = 0; which < 2; which++) {
    const std::vector<float>& depth = *pictures[which];
    for (int k = 0; k < numBoxes; k++) {
      gray[k] = 0;
      if (depth[k] < 1e29f) {
        float s = (depth[k] - dMin) / (dMax - dMin);   // where this distance sits in the range: 0 to 1
        s = fminf(fmaxf(s, 0.0f), 1.0f);               // outside the range: pinned to 0 or 1
        gray[k] = (unsigned char)(255 * (1 - s));      // 0 -> white (255), 1 -> black (0)
      }
    }
    if (writePGM(names[which], gray.data(), PW, PH)) printf("saved %s\n", names[which]);
    else printf("could not save %s\n", names[which]);
  }

  return 0;
}
