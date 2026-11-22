// The interactive viewer's drawing program. It runs on Colab and never exits on
// its own: it waits for a line of text saying how the camera should move, draws
// one picture with the two step 2 kernels, sends the picture back, and waits again.
// The web server (viewer/server.py) is the one talking to it.
//
// HOW IT TALKS (all through standard input and standard output)
//   On start it prints one line:      POINTWALK <W> <H> <number of dots>
//   Then, for every line it receives:
//       <forward> <right> <up> <turn> <look>     five numbers, see src/move.h
//       reset                                    put the camera back at the start
//   it answers with one line:         FRAME <x> <y> <z> <yaw> <pitch> <kernel ms> <frame ms>
//   followed by W * H * 3 raw bytes:  the picture, r g b per box, top row first.
//   Anything meant for a human goes to standard error, so it cannot corrupt a picture.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include "camera.h"     // Camera, setPose
#include "image.h"      // Color
#include "pack.h"       // Packed, packDepthIndex
#include "point.h"      // Point
#include "kernels.h"    // colorKernel, resolveKernel
#include "scenefile.h"  // readScene
#include "move.h"       // Pose, applyMove

// Stop with a clear message if any CUDA call fails (CUDA errors are silent otherwise)
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// Same lens as project.cu
const int W = 640, H = 480;
const float fx = 500.0f, fy = 500.0f, cx = 320.0f, cy = 240.0f;

int main(int argc, char** argv) {
  const char* scenePath = (argc > 1) ? argv[1] : "scene.bin";

  // ---- Load the dots and colours that project.cu saved ----
  std::vector<Point> pts;
  std::vector<Color> colors;
  if (!readScene(scenePath, pts, colors)) {
    fprintf(stderr, "viewer: could not read the scene file %s (run ./project first)\n", scenePath);
    return 1;
  }
  const int N = (int)pts.size();

  // ---- Send the scene to the GPU once. It stays there for every picture. ----
  Point* d_pts;
  Color* d_colors;
  Packed* d_notes;        // one note per box, rewritten for every picture
  Packed* d_emptyNotes;   // a spare set of empty notes, used to wipe d_notes quickly
  Color* d_image;
  CHECK(cudaMalloc(&d_pts, N * sizeof(Point)));
  CHECK(cudaMalloc(&d_colors, N * sizeof(Color)));
  CHECK(cudaMalloc(&d_notes, W * H * sizeof(Packed)));
  CHECK(cudaMalloc(&d_emptyNotes, W * H * sizeof(Packed)));
  CHECK(cudaMalloc(&d_image, W * H * sizeof(Color)));
  CHECK(cudaMemcpy(d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_colors, colors.data(), N * sizeof(Color), cudaMemcpyHostToDevice));

  const Packed emptyNote = packDepthIndex(1e30f, 0);   // "really far", same as project.cu
  std::vector<Packed> emptyNotes(W * H, emptyNote);
  CHECK(cudaMemcpy(d_emptyNotes, emptyNotes.data(), W * H * sizeof(Packed), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int dotBlocks = (N + threads - 1) / threads;       // enough blocks to cover all N dots
  const int boxBlocks = (W * H + threads - 1) / threads;   // enough blocks to cover all W*H boxes

  cudaEvent_t kernelStart, kernelEnd;                      // stopwatch for the two kernels
  CHECK(cudaEventCreate(&kernelStart));
  CHECK(cudaEventCreate(&kernelEnd));

  // ---- The camera: fixed lens, movable pose ----
  Camera cam = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}, fx, fy, cx, cy, W, H};
  const Pose startPose = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};   // at the origin, facing +z, level
  Pose pose = startPose;

  Color* image;                                           // the picture, on the CPU side
  CHECK(cudaMallocHost(&image, W * H * sizeof(Color)));   // locked memory: the GPU can copy straight into it

  // ---- Say hello, then answer one line at a time until the input closes ----
  printf("POINTWALK %d %d %d\n", W, H, N);
  fflush(stdout);
  fprintf(stderr, "viewer: ready, %d dots\n", N);

  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    if (strncmp(line, "quit", 4) == 0) break;

    if (strncmp(line, "reset", 5) == 0) {
      pose = startPose;
    } else {
      float forward, right, up, turn, look;
      if (sscanf(line, "%f %f %f %f %f", &forward, &right, &up, &turn, &look) == 5) {
        applyMove(pose, forward, right, up, turn, look);   // your function, in move.h
      }
      // A line that is not five numbers moves nothing, but still gets a picture back,
      // so the two sides never lose count of whose turn it is.
    }

    auto frameStart = std::chrono::steady_clock::now();
    setPose(cam, pose.x, pose.y, pose.z, pose.yaw, pose.pitch);

    // Wipe the notes: copy the spare empty set over them (GPU to GPU, no trip to the CPU).
    CHECK(cudaMemcpy(d_notes, d_emptyNotes, W * H * sizeof(Packed), cudaMemcpyDeviceToDevice));

    CHECK(cudaEventRecord(kernelStart));
    colorKernel<<<dotBlocks, threads>>>(d_pts, N, cam, d_notes);             // pass 1: notes
    resolveKernel<<<boxBlocks, threads>>>(d_notes, d_colors, W * H, d_image); // pass 2: picture
    CHECK(cudaEventRecord(kernelEnd));
    CHECK(cudaEventSynchronize(kernelEnd));
    CHECK(cudaGetLastError());
    float kernelMs = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelMs, kernelStart, kernelEnd));

    CHECK(cudaMemcpy(image, d_image, W * H * sizeof(Color), cudaMemcpyDeviceToHost));
    float frameMs = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - frameStart).count();

    printf("FRAME %f %f %f %f %f %f %f\n", pose.x, pose.y, pose.z, pose.yaw, pose.pitch, kernelMs, frameMs);
    fwrite(image, sizeof(Color), W * H, stdout);
    fflush(stdout);
  }

  cudaEventDestroy(kernelStart);
  cudaEventDestroy(kernelEnd);
  cudaFree(d_pts);
  cudaFree(d_colors);
  cudaFree(d_notes);
  cudaFree(d_emptyNotes);
  cudaFree(d_image);
  cudaFreeHost(image);
  return 0;
}
