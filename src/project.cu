#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "camera.h"    // the Camera struct and setPose

// Camera parameters
const int W = 640;
const int H = 480;
const float fx = 500.0f, fy = 500.0f;    // focal length (zoom), in pixels
const float cx = 320.0f, cy = 240.0f;    // image center
const float R[3][3] = {{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}};
const float t[3] = {0, 0, 0};

struct Point {
  float x, y , z;
};

__device__ void atomicMinFloat(float*address, float val) {
    atomicMin((int*)address, __float_as_int(val));
}

// Second version, copied unchanged from src/step0.cu. Not called here; bench.cu times both.
__device__ void atomicMinFloatGeneric(float*address, float val) {
    int *a = (int*)address;
    int old = *a;
    while (val < __int_as_float(old)) {
        int prev = atomicCAS(a, old, __float_as_int(val));
        if (prev == old) break;
        old = prev;
    }
}


// Runs on the GPU. One thread = one dot. Same logic as your CPU loop body.
__global__ void depthKernel(const Point* pts, int N, Camera cam, float* depth) {
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
  atomicMinFloat(&depth[idx], zc);                 // closer? overwrite the note

}
float randFloat(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

// CHANGED: added `float& z` so project also hands back the camera depth
bool project(Point p, float& u, float& v, float& z) {
  float xc = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z + t[0];
  float yc = R[1][0]*p.x + R[1][1]*p.y + R[1][2]*p.z + t[1];
  float zc = R[2][0]*p.x + R[2][1]*p.y + R[2][2]*p.z + t[2];
  if (zc < 0.1f) return false;
  u = fx * (xc / zc) + cx;
  v = fy * (yc / zc) + cy;
  z = zc;                                  // NEW: depth as seen from the camera
  return true;
}

int main() {
  // ---- Camera: the lens numbers, then where it stands and which way it faces ----
  Camera cam = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}, fx, fy, cx, cy, W, H};
  setPose(cam, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);   // position x, y, z in metres, then yaw, pitch in radians

  const int N = 1000000;
  const float PI = 3.14159265f;
  std::vector<Point> pts(N);

  // ---- Scene settings ----
  const float floorY = 1.5f;
  const float radius = 2.0f;
  const Point center = {0.0f, floorY - radius, 8.0f};

  // ---- Generate points ----
  srand(0);
  for (int i = 0; i < N; i++) {
    if (i < N / 2) {
      pts[i].x = randFloat(-10, 10);
      pts[i].y = floorY;
      pts[i].z = randFloat(1, 30);
    }
    else {
      float h = randFloat(-1, 1);
      float theta = randFloat(0, 2 * PI);
      float s = sqrtf(1 - h * h);
      pts[i].x = s * cosf(theta) * radius + center.x;
      pts[i].y = h * radius + center.y;
      pts[i].z = s * sinf(theta) * radius + center.z;
    }
  }

  // ---- Check sphere ----
  float maxErr = 0.0f;
  for (int i = N / 2; i < N; i++) {
    float dx = pts[i].x - center.x;
    float dy = pts[i].y - center.y;
    float dz = pts[i].z - center.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float err = fabsf(dist - radius);
    maxErr = fmaxf(maxErr, err);
  }
  printf("max sphere error: %e\n", maxErr);

  // ---- Save points for the 3D viewer ----
  FILE* f = fopen("points.txt", "w");
  for (int i = 0; i < N; i++) {
    fprintf(f, "%f %f %f\n", pts[i].x, pts[i].y, pts[i].z);
  }
  fclose(f);

  // ---- Save projected pixels for the scatter viewer ----
  FILE* img = fopen("pixels.txt", "w");
  int visible = 0;
  for (int i = 0; i < N; i++) {
    float u, v, z;                                   // CHANGED: added z
    if (project(pts[i], u, v, z)) {                  // CHANGED: pass z
      if (u >= 0 && u < W && v >= 0 && v < H) {
        fprintf(img, "%f %f %f\n", u, v, z);         // CHANGED: camera depth, not pts[i].z
        visible++;
      }
    }
  }
  fclose(img);
  printf("%d of %d dots are visible in the photo\n", visible, N);

  // ---- NEW: Depth buffer — one sticky note per pixel, keep the closest ----
  std::vector<float> depth(W * H, 1e30f);            // every pixel starts "really far"
  for (int i = 0; i < N; i++) {
    float u, v, z;
    if (!project(pts[i], u, v, z)) continue;         // behind camera → skip

    int px = (int)floorf(u);                         // which column box
    int py = (int)floorf(v);                         // which row box
    if (px < 0 || px >= W || py < 0 || py >= H) continue;   // off the photo → skip

    int idx = py * W + px;                           // box position in the long list
    if (z < depth[idx]) depth[idx] = z;              // closer? overwrite the note
  }

  // ---- NEW: Count how many boxes got something ----
  int filled = 0;
  for (int i = 0; i < W * H; i++) {
    if (depth[i] < 1e29f) filled++;
  }
  printf("%d of %d pixels have something in them\n", filled, W * H);

  // ---- NEW: Save depth as a grayscale image (near = white, far = dark, empty = black) ----
  const float dMin = 1.0f, dMax = 30.0f;
  std::vector<unsigned char> gray(W * H, 0);
  for (int i = 0; i < W * H; i++) {
    if (depth[i] < 1e29f) {
      float s = (depth[i] - dMin) / (dMax - dMin);   // where this distance sits in the range: 0 to 1
      s = fminf(fmaxf(s, 0.0f), 1.0f);               // anything outside the range gets pinned to 0 or 1
      gray[i] = 255 * (1 - s);                       // 0 → white (255), 1 → black (0)
    }
  }

  // ---- GPU depth buffer (uses the camera set up at the top of main) ----
  Point* d_pts;      // "d_" = lives on the device (GPU)
  float* d_depth;
  cudaMalloc(&d_pts, N * sizeof(Point));               // like malloc, but on the GPU
  cudaMalloc(&d_depth, W * H * sizeof(float));

  std::vector<float> gpuDepth(W * H, 1e30f);           // starting values: "really far"
  cudaMemcpy(d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(d_depth, gpuDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;            // enough blocks to cover all N dots
  depthKernel<<<blocks, threads>>>(d_pts, N, cam, d_depth);
  cudaDeviceSynchronize();                              // wait for all threads (like pthread_join)
  printf("kernel status: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(gpuDepth.data(), d_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_pts);
  cudaFree(d_depth);

  int mismatches = 0;
  for (int k = 0; k < W * H; k++) {
    if (gpuDepth[k] != depth[k]) mismatches++;
  }
  printf("GPU vs CPU: %d of %d pixels differ\n", mismatches, W * H);


  FILE* pg = fopen("depth.pgm", "wb");
  fprintf(pg, "P5\n%d %d\n255\n", W, H);
  fwrite(gray.data(), 1, W * H, pg);
  fclose(pg);
  printf("saved depth.pgm\n");

  return 0;


}
