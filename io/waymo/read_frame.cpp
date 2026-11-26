// Tiny check of the range image reader: read ONE converted frame and print its
// H, W, C and its first few ranges, to compare by eye with what
// waymo/inspect_segment.py prints for the same frame and laser.
//
// Build and run, from the repo folder (no GPU needed):
//   g++ -std=c++17 -O2 -Wall -Wextra io/waymo/read_frame.cpp -o build/read_frame
//   ./build/read_frame <out>/range/frame0000_laser1.bin
//
// <out> is the folder given to waymo/convert.py. Frame 0000 is the first
// frame, which is the one inspect_segment.py prints values for.
//
// This is not in tests/ because tests/run.sh runs everything there with no
// arguments, and this needs a converted file.

#include <cstdio>
#include "waymofile.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <out>/range/frameFFFF_laserL.bin\n", argv[0]);
    return 1;
  }

  WaymoRangeImage image;
  if (!readWaymoRange(argv[1], image)) return 1;   // the reader already said why

  printf("frame %d, laser %d, timestamp %lld\n", image.frame, image.laser, image.timestamp);
  printf("shape [H, W, C] = [%d, %d, %d]\n", image.H, image.W, image.C);

  // The range is channel 0. These are the first 8 boxes of row 0, raw.
  int shown = image.W < 8 ? image.W : 8;
  printf("first %d ranges (channel 0, row 0, columns 0 to %d):", shown, shown - 1);
  for (int col = 0; col < shown; col++) printf(" %.4f", image.values[(size_t)col * image.C]);
  printf("\n");

  // Row 0 often starts with boxes that have no return (-1), so also show the
  // first 8 boxes that DO have one, going through the file in its own order.
  size_t boxes = (size_t)image.H * image.W;
  printf("first ranges with a return, as row,col=range:");
  int found = 0;
  for (size_t box = 0; box < boxes && found < 8; box++) {
    float range = image.values[box * image.C];
    if (range > 0.0f) {
      printf(" %d,%d=%.4f", (int)(box / image.W), (int)(box % image.W), range);
      found++;
    }
  }
  printf("\n");

  // Same two counts as inspect_segment.py prints for channel 0.
  long long withReturn = 0, noReturn = 0;
  for (size_t box = 0; box < boxes; box++) {
    float range = image.values[box * image.C];
    if (range > 0.0f) withReturn++;
    if (range == -1.0f) noReturn++;
  }
  printf("channel 0: boxes > 0: %lld, boxes == -1: %lld\n", withReturn, noReturn);
  return 0;
}
