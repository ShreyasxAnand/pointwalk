#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>

// Stop with a clear message if any CUDA call fails (CUDA errors are silent otherwise)
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

const int W = 640, H = 480;
const float fx = 500.0f, fy = 500.0f, cx = 320.0f, cy = 240.0f;

struct Point { float x, y, z; };
struct Camera { float R[3][3]; float t[3]; float fx, fy, cx, cy; int W, H; };

float randFloat(float a, float b) { return a + (b - a) * ((float)rand() / RAND_MAX); }

// Same scene as project.cu: half floor, half ball resting on it
void makeScene(std::vector<Point>& pts) {
  const int N = pts.size();
  const float PI = 3.14159265f, floorY = 1.5f, radius = 2.0f;
  const Point center = {0.0f, floorY - radius, 8.0f};
  srand(0);
  for (int i = 0; i < N; i++) {
    if (i < N / 2) {
      pts[i].x = randFloat(-10, 10);
      pts[i].y = floorY;
      pts[i].z = randFloat(1, 30);
    } else {
      float h = randFloat(-1, 1), theta = randFloat(0, 2 * PI), s = sqrtf(1 - h * h);
      pts[i].x = s * cosf(theta) * radius + center.x;
      pts[i].y = h * radius + center.y;
      pts[i].z = s * sinf(theta) * radius + center.z;
    }
  }
}

// CPU reference: one worker, one dot at a time
void cpuDepth(const std::vector<Point>& pts, const Camera& cam, std::vector<float>& depth) {
  std::fill(depth.begin(), depth.end(), 1e30f);
  for (size_t i = 0; i < pts.size(); i++) {
    const Point& p = pts[i];
    float xc = cam.R[0][0]*p.x + cam.R[0][1]*p.y + cam.R[0][2]*p.z + cam.t[0];
    float yc = cam.R[1][0]*p.x + cam.R[1][1]*p.y + cam.R[1][2]*p.z + cam.t[1];
    float zc = cam.R[2][0]*p.x + cam.R[2][1]*p.y + cam.R[2][2]*p.z + cam.t[2];
    if (zc < 0.1f) continue;
    int px = (int)floorf(cam.fx * (xc / zc) + cam.cx);
    int py = (int)floorf(cam.fy * (yc / zc) + cam.cy);
    if (px < 0 || px >= cam.W || py < 0 || py >= cam.H) continue;
    int idx = py * cam.W + px;
    if (zc < depth[idx]) depth[idx] = zc;
  }
}

// Copied unchanged from src/step0.cu: two ways to keep the smallest depth in a cell.
__device__ void atomicMinFloat(float*address, float val) {
    atomicMin((int*)address, __float_as_int(val));
}

__device__ void atomicMinFloatGeneric(float*address, float val) {
    int *a = (int*)address;
    int old = *a;
    while (val < __int_as_float(old)) {
        int prev = atomicCAS(a, old, __float_as_int(val));
        if (prev == old) break;
        old = prev;
    }
}

// GPU version: one thread per dot. useLoop picks which of the two functions above
// fixes the race: false = atomicMinFloat, true = atomicMinFloatGeneric.
__global__ void depthKernel(const Point* pts, int N, Camera cam, float* depth, bool useLoop) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  Point p = pts[i];
  float xc = cam.R[0][0]*p.x + cam.R[0][1]*p.y + cam.R[0][2]*p.z + cam.t[0];
  float yc = cam.R[1][0]*p.x + cam.R[1][1]*p.y + cam.R[1][2]*p.z + cam.t[1];
  float zc = cam.R[2][0]*p.x + cam.R[2][1]*p.y + cam.R[2][2]*p.z + cam.t[2];
  if (zc < 0.1f) return;
  int px = (int)floorf(cam.fx * (xc / zc) + cam.cx);
  int py = (int)floorf(cam.fy * (yc / zc) + cam.cy);
  if (px < 0 || px >= cam.W || py < 0 || py >= cam.H) return;
  float* cell = &depth[py * cam.W + px];
  if (useLoop) atomicMinFloatGeneric(cell, zc);
  else         atomicMinFloat(cell, zc);
}

// Time the kernel alone, REPS times, and keep the fastest run. The depth buffer is
// reset to "really far" before every run (not timed), so every run does the full work.
// After it returns, d_depth holds the result of the last run.
float bestKernelMs(const Point* d_pts, int N, Camera cam, float* d_depth,
                   const std::vector<float>& farDepth, bool useLoop) {
  const int REPS = 20;
  int threads = 256, blocks = (N + threads - 1) / threads;
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  float best = 1e30f;
  for (int r = 0; r < REPS; r++) {
    CHECK(cudaMemcpy(d_depth, farDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(a));
    depthKernel<<<blocks, threads>>>(d_pts, N, cam, d_depth, useLoop);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return best;
}

int main() {
  Camera cam = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}, fx, fy, cx, cy, W, H};
  CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));   // allow the GPU to read mapped CPU memory
  CHECK(cudaFree(0));   // wake up the GPU now, so startup cost isn't counted in any timing

  int sizes[3] = {100000, 1000000, 10000000};
  float minMs[3], loopMs[3];                 // kernel-only times for the two versions
  int minDiff[3] = {0}, loopDiff[3] = {0};   // pixels that differ from the CPU image
  printf("%10s %10s %11s %11s %14s %6s %6s\n",
       "N", "CPU ms", "kernel ms", "pinned ms", "zero-copy ms", "diff", "zdiff");

  for (int s = 0; s < 3; s++) {
    int N = sizes[s];
    std::vector<Point> pts(N);
    makeScene(pts);
    std::vector<float> cpu(W * H), gpu(W * H, 1e30f);

    // ---- CPU timing ----
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuDepth(pts, cam, cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- GPU setup (not timed) ----
    Point* d_pts; float* d_depth;
    CHECK(cudaMalloc(&d_pts, N * sizeof(Point)));
    CHECK(cudaMalloc(&d_depth, W * H * sizeof(float)));
    int threads = 256, blocks = (N + threads - 1) / threads;
    Point* h_pts;                                         // pinned CPU buffer
    CHECK(cudaMallocHost(&h_pts, N * sizeof(Point)));     // allocate locked memory
    std::copy(pts.begin(), pts.end(), h_pts);             // copy points into it

    // Warm-up run (not timed): the first launch of a kernel is slower
    CHECK(cudaMemcpy(d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_depth, gpu.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    depthKernel<<<blocks, threads>>>(d_pts, N, cam, d_depth, false);
    CHECK(cudaDeviceSynchronize());

    // ---- GPU timing: copies in, kernel, copy out ----
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

    CHECK(cudaEventRecord(e0));                               // start: before copying in
    CHECK(cudaMemcpy(d_pts, h_pts, N * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_depth, gpu.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(e1));                               // kernel starts
    depthKernel<<<blocks, threads>>>(d_pts, N, cam, d_depth, false);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e2));                               // kernel ends
    CHECK(cudaMemcpy(gpu.data(), d_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(e3));                               // end: after copying out
    CHECK(cudaEventSynchronize(e3));

    float kernelMs, totalMs;
    cudaEventElapsedTime(&kernelMs, e1, e2);
    cudaEventElapsedTime(&totalMs, e0, e3);

    // ---- Zero-copy: kernel reads points straight from CPU RAM over PCIe ----
    Point* h_mapped;                                           // CPU-side pointer
    Point* d_mapped;                                           // GPU-side pointer to the SAME memory
    CHECK(cudaHostAlloc(&h_mapped, N * sizeof(Point), cudaHostAllocMapped));
    std::copy(pts.begin(), pts.end(), h_mapped);
    CHECK(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));
    std::vector<float> zeroDepth(W * H, 1e30f);

    // Warm-up (not timed)
    CHECK(cudaMemcpy(d_depth, zeroDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    depthKernel<<<blocks, threads>>>(d_mapped, N, cam, d_depth, false);
    CHECK(cudaDeviceSynchronize());

    // Timed run: NO cudaMemcpy for the points
    cudaEvent_t z0, z1;
    cudaEventCreate(&z0); cudaEventCreate(&z1);
    CHECK(cudaEventRecord(z0));
    CHECK(cudaMemcpy(d_depth, zeroDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    depthKernel<<<blocks, threads>>>(d_mapped, N, cam, d_depth, false);   // reads points over PCIe
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(zeroDepth.data(), d_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(z1));
    CHECK(cudaEventSynchronize(z1));

    float zeroMs;
    cudaEventElapsedTime(&zeroMs, z0, z1);

    int zdiff = 0;
    for (int k = 0; k < W * H; k++) if (zeroDepth[k] != cpu[k]) zdiff++;

    // ---- Kernel only, both versions, best of 20 runs each ----
    std::vector<float> farDepth(W * H, 1e30f), out(W * H);
    minMs[s] = bestKernelMs(d_pts, N, cam, d_depth, farDepth, false);
    CHECK(cudaMemcpy(out.data(), d_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < W * H; k++) if (out[k] != cpu[k]) minDiff[s]++;
    loopMs[s] = bestKernelMs(d_pts, N, cam, d_depth, farDepth, true);
    CHECK(cudaMemcpy(out.data(), d_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < W * H; k++) if (out[k] != cpu[k]) loopDiff[s]++;

    // ---- Correctness: GPU must match CPU exactly ----
    int diff = 0;
    for (int k = 0; k < W * H; k++) if (gpu[k] != cpu[k]) diff++;

    printf("%10d %10.2f %11.3f %11.2f %14.2f %6d %6d\n",
       N, cpuMs, kernelMs, totalMs, zeroMs, diff, zdiff);

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(z0); cudaEventDestroy(z1);
    cudaFreeHost(h_mapped);
    cudaFree(d_pts); cudaFree(d_depth); cudaFreeHost(h_pts);
  }

  printf("\nKernel only, fastest of 20 runs. diff = pixels that differ from the CPU image.\n");
  printf("%10s %14s %6s %14s %6s\n", "N", "atomicMin ms", "diff", "CAS loop ms", "diff");
  for (int s = 0; s < 3; s++)
    printf("%10d %14.3f %6d %14.3f %6d\n", sizes[s], minMs[s], minDiff[s], loopMs[s], loopDiff[s]);
  return 0;
}
