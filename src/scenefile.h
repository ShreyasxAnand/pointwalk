#ifndef SCENEFILE_H
#define SCENEFILE_H

// The scene file: how project.cu hands the dots to the viewer.
// Plain C++, no CUDA in here.
//
// Layout of the file, in order:
//   8 bytes   the letters "PWSCENE1" (so a wrong file is noticed straight away)
//   4 bytes   N, the number of dots (int)
//   N * 12    the dots:    x, y, z as three floats each
//   N * 3     the colours: r, g, b as three bytes each, same numbering as the dots

#include <cstdio>
#include <cstring>
#include <vector>
#include "point.h"   // Point
#include "image.h"   // Color

// Save the scene. Returns false if the file could not be written.
inline bool writeScene(const char* path, const std::vector<Point>& pts, const std::vector<Color>& colors) {
  if (pts.size() != colors.size()) return false;      // one colour per dot, or the file makes no sense
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  int n = (int)pts.size();
  bool ok = fwrite("PWSCENE1", 1, 8, f) == 8
         && fwrite(&n, sizeof(int), 1, f) == 1
         && fwrite(pts.data(), sizeof(Point), pts.size(), f) == pts.size()
         && fwrite(colors.data(), sizeof(Color), colors.size(), f) == colors.size();
  return (fclose(f) == 0) && ok;
}

// Load the scene. Returns false if the file is missing, cut short, or not a scene file.
inline bool readScene(const char* path, std::vector<Point>& pts, std::vector<Color>& colors) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[8];
  int n = 0;
  bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "PWSCENE1", 8) == 0
         && fread(&n, sizeof(int), 1, f) == 1 && n > 0;
  if (ok) {
    pts.resize(n);
    colors.resize(n);
    ok = fread(pts.data(), sizeof(Point), pts.size(), f) == pts.size()
      && fread(colors.data(), sizeof(Color), colors.size(), f) == colors.size();
  }
  fclose(f);
  return ok;
}

#endif
