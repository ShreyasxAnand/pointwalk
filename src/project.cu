#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "camera.h"    // the Camera struct and setPose
#include "image.h"     // Color and writePPM
#include "pack.h"      // packDepthIndex, unpackDepth, unpackIndex

// Camera parameters
const int W = 640;
const int H = 480;
const float fx = 500.0f, fy = 500.0f;    // focal length (zoom), in pixels
const float cx = 320.0f, cy = 240.0f;    // image center

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

float randFloat(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

// CHANGED: added `float& z` so project also hands back the camera depth
// STEP 1: added `cam`, so the CPU uses the same R and t as the GPU kernel
bool project(Point p, const Camera& cam, float& u, float& v, float& z) {
  float xc = cam.R[0][0]*p.x + cam.R[0][1]*p.y + cam.R[0][2]*p.z + cam.t[0];
  float yc = cam.R[1][0]*p.x + cam.R[1][1]*p.y + cam.R[1][2]*p.z + cam.t[1];
  float zc = cam.R[2][0]*p.x + cam.R[2][1]*p.y + cam.R[2][2]*p.z + cam.t[2];
  if (zc < 0.1f) return false;
  u = fx * (xc / zc) + cx;
  v = fy * (yc / zc) + cy;
  z = zc;                                  // NEW: depth as seen from the camera
  return true;
}

int main() {
  // ---- Camera: the lens numbers, then where it stands and which way it faces ----
  Camera cam = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}, fx, fy, cx, cy, W, H};
  setPose(cam, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f);  // position x, y, z in metres, then yaw, pitch in radians

  const int N = 1000000;
  const float PI = 3.14159265f;
  std::vector<Point> pts(N);
  std::vector<Color> colors(N);          // colors[i] is the colour of dot i (same numbering as pts)
  const Color floorColor = {170, 170, 170};   // grey
  const Color ballColor  = {220, 40, 40};     // red

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
      colors[i] = floorColor;
    }
    else {
      float h = randFloat(-1, 1);
      float theta = randFloat(0, 2 * PI);
      float s = sqrtf(1 - h * h);
      pts[i].x = s * cosf(theta) * radius + center.x;
      pts[i].y = h * radius + center.y;
      pts[i].z = s * sinf(theta) * radius + center.z;
      colors[i] = ballColor;
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
    if (project(pts[i], cam, u, v, z)) {             // CHANGED: pass z
      if (u >= 0 && u < W && v >= 0 && v < H) {
        fprintf(img, "%f %f %f\n", u, v, z);         // CHANGED: camera depth, not pts[i].z
        visible++;
      }
    }
  }
  fclose(img);
  printf("%d of %d dots are visible in the photo\n", visible, N);

  // ---- Step 1 check: stepping back 1 m must add exactly 1 to every depth ----
  // Compare each dot's depth from this camera with its depth from a camera that
  // has not moved. Only meaningful while the pose above is "back 1 m, no turning".
  Camera still = cam;
  setPose(still, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  int compared = 0, exact = 0;
  for (int i = 0; i < N; i++) {
    float u, v, zStill, zMoved;
    if (!project(pts[i], still, u, v, zStill)) continue;   // behind the unmoved camera
    if (!project(pts[i], cam, u, v, zMoved)) continue;     // behind the moved camera
    compared++;
    if (zMoved == zStill + 1.0f) exact++;
  }
  printf("back 1 m: %d of %d dots are exactly 1 m deeper\n", exact, compared);

  // An empty note says "really far" (same 1e30 as the depth buffer), so any real
  // dot is smaller and replaces it. Used by both the CPU notes and the GPU notes.
  const Packed emptyNote = packDepthIndex(1e30f, 0);

  // ---- NEW: Depth buffer — one sticky note per pixel, keep the closest ----
  std::vector<float> depth(W * H, 1e30f);            // every pixel starts "really far"
  for (int i = 0; i < N; i++) {
    float u, v, z;
    if (!project(pts[i], cam, u, v, z)) continue;    // behind camera → skip

    int px = (int)floorf(u);                         // which column box
    int py = (int)floorf(v);                         // which row box
    if (px < 0 || px >= W || py < 0 || py >= H) continue;   // off the photo → skip

    int idx = py * W + px;                           // box position in the long list
    if (z < depth[idx]) depth[idx] = z;              // closer? overwrite the note
  }

  // ---- Step 2: CPU notes. Same loop as above, but each note holds (distance, dot number) glued ----
  // This is the slow, obviously-right version that the GPU notes get compared against.
  std::vector<Packed> cpuNotes(W * H, emptyNote);    // every note starts empty
  for (int i = 0; i < N; i++) {
    float u, v, z;
    if (!project(pts[i], cam, u, v, z)) continue;    // behind camera → skip

    int px = (int)floorf(u);                         // which column box
    int py = (int)floorf(v);                         // which row box
    if (px < 0 || px >= W || py < 0 || py >= H) continue;   // off the photo → skip

    int idx = py * W + px;                           // box position in the long list
    Packed glued = packDepthIndex(z, i);             // glue: distance on the left, dot number on the right
    if (glued < cpuNotes[idx]) cpuNotes[idx] = glued;   // smaller = closer? overwrite the note
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
  cudaFree(d_depth);

  // ---- Step 2: GPU notes. One per box, each holding (distance, dot number) glued together ----
  std::vector<Packed> gpuNotes(W * H, emptyNote);      // starting values: every note empty
  Packed* d_notes;
  cudaMalloc(&d_notes, W * H * sizeof(Packed));
  cudaMemcpy(d_notes, gpuNotes.data(), W * H * sizeof(Packed), cudaMemcpyHostToDevice);

  colorKernel<<<blocks, threads>>>(d_pts, N, cam, d_notes);   // same dots, same camera, new notes
  cudaDeviceSynchronize();
  printf("color kernel status: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(gpuNotes.data(), d_notes, W * H * sizeof(Packed), cudaMemcpyDeviceToHost);
  cudaFree(d_pts);

  // ---- Step 2: second pass. Turn the notes into a picture, one thread per box ----
  // The notes stay on the GPU from the first pass. The colour list has to be sent up.
  Color* d_colors;
  Color* d_image;
  cudaMalloc(&d_colors, N * sizeof(Color));
  cudaMalloc(&d_image, W * H * sizeof(Color));
  cudaMemcpy(d_colors, colors.data(), N * sizeof(Color), cudaMemcpyHostToDevice);

  int boxBlocks = (W * H + threads - 1) / threads;     // enough blocks to cover all W*H boxes
  resolveKernel<<<boxBlocks, threads>>>(d_notes, d_colors, W * H, d_image);
  cudaDeviceSynchronize();
  printf("resolve kernel status: %s\n", cudaGetErrorString(cudaGetLastError()));

  std::vector<Color> gpuImage(W * H);
  cudaMemcpy(gpuImage.data(), d_image, W * H * sizeof(Color), cudaMemcpyDeviceToHost);
  cudaFree(d_notes);
  cudaFree(d_colors);
  cudaFree(d_image);

  if (writePPM("color.ppm", gpuImage.data(), W, H)) printf("saved color.ppm\n");
  else printf("could not save color.ppm\n");

  // How many notes got a dot, and what the note in the middle of the photo says.
  int notesUsed = 0;
  for (int k = 0; k < W * H; k++) {
    if (unpackDepth(gpuNotes[k]) < 1e29f) notesUsed++;
  }
  printf("%d of %d notes have a dot on them\n", notesUsed, W * H);

  Packed middle = gpuNotes[(H / 2) * W + (W / 2)];
  if (unpackDepth(middle) < 1e29f)
    printf("middle box: dot %u, %f m away\n", unpackIndex(middle), unpackDepth(middle));
  else
    printf("middle box: empty\n");

  int mismatches = 0;
  for (int k = 0; k < W * H; k++) {
    if (gpuDepth[k] != depth[k]) mismatches++;
  }
  printf("GPU vs CPU: %d of %d pixels differ\n", mismatches, W * H);

  // Step 2 check: every GPU note must equal the CPU note for the same box.
  // Comparing the whole glued number checks the distance AND the dot number at once.
  int noteMismatches = 0;
  for (int k = 0; k < W * H; k++) {
    if (gpuNotes[k] != cpuNotes[k]) noteMismatches++;
  }
  printf("GPU vs CPU notes: %d of %d differ\n", noteMismatches, W * H);

  FILE* pg = fopen("depth.pgm", "wb");
  fprintf(pg, "P5\n%d %d\n255\n", W, H);
  fwrite(gray.data(), 1, W * H, pg);
  fclose(pg);
  printf("saved depth.pgm\n");

  return 0;


}
