#ifndef IMAGE_H
#define IMAGE_H

// Colours and the colour image file. Plain C++, no CUDA in here.

#include <cstdio>

// One colour: red, green, blue, each from 0 (none) to 255 (full).
struct Color { unsigned char r, g, b; };

// Save W x H colours as a .ppm picture (row 0 is the top row).
// Returns false if the file could not be opened.
inline bool writePPM(const char* path, const Color* pixels, int W, int H) {
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  fprintf(f, "P6\n%d %d\n255\n", W, H);
  fwrite(pixels, sizeof(Color), (size_t)W * H, f);
  fclose(f);
  return true;
}

#endif
