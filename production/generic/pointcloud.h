#ifndef POINTWALK_GENERIC_POINTCLOUD_H
#define POINTWALK_GENERIC_POINTCLOUD_H

// Point-cloud file: dots and one colour per dot. The label stays "PWSCENE1"
// so the viewer can read it. On failure, both functions print the path and
// the reason to stderr, then return false.
//
// Layout (little-endian):
//   8 bytes    "PWSCENE1", no trailing '\0'
//   4 bytes    N, the number of dots (int, at least 1)
//   N * 12     dots:    x, y, z floats    (Point)
//   N * 3      colours: r, g, b bytes     (Color); colour i belongs to dot i

#include <climits>   // INT_MAX
#include <cstdio>
#include <cstring>   // memcmp
#include <vector>
#include "color.h"   // Color
#include "point.h"   // Point

const char pointCloudLabel[] = "PWSCENE1";
const size_t pointCloudLabelBytes = sizeof(pointCloudLabel) - 1;   // 8, without the '\0'
const long long pointCloudHeaderBytes = pointCloudLabelBytes + sizeof(int);
const long long pointCloudBytesPerDot = sizeof(Point) + sizeof(Color);

// Write the cloud to path. Refuses zero dots, so it never writes a file
// readPointCloud would reject.
inline bool writePointCloud(const char* path, const std::vector<Point>& pts,
                            const std::vector<Color>& colors) {
  if (pts.size() != colors.size()) {
    fprintf(stderr, "%s: %zu dots but %zu colours, need one colour per dot\n",
            path, pts.size(), colors.size());
    return false;
  }
  if (pts.empty()) {
    fprintf(stderr, "%s: no dots, an empty point cloud is not written\n", path);
    return false;
  }
  if (pts.size() > static_cast<size_t>(INT_MAX)) {
    fprintf(stderr, "%s: %zu dots, more than the file's count can hold (%d)\n",
            path, pts.size(), INT_MAX);
    return false;
  }

  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "%s: cannot open for writing\n", path);
    return false;
  }
  const int n = static_cast<int>(pts.size());
  bool ok = fwrite(pointCloudLabel, 1, pointCloudLabelBytes, f) == pointCloudLabelBytes
         && fwrite(&n, sizeof(int), 1, f) == 1
         && fwrite(pts.data(), sizeof(Point), pts.size(), f) == pts.size()
         && fwrite(colors.data(), sizeof(Color), colors.size(), f) == colors.size();
  if (fclose(f) != 0) ok = false;
  if (!ok) fprintf(stderr, "%s: write failed\n", path);
  return ok;
}

// Read path into pts and colors. Checks the label, the count and the file
// size before allocating; on failure both are left empty.
inline bool readPointCloud(const char* path, std::vector<Point>& pts,
                           std::vector<Color>& colors) {
  pts.clear();
  colors.clear();

  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "%s: cannot open for reading\n", path);
    return false;
  }
  long long fileBytes = -1;
  if (fseek(f, 0, SEEK_END) == 0) fileBytes = static_cast<long long>(ftell(f));
  if (fileBytes < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fprintf(stderr, "%s: cannot find the file's size\n", path);
    fclose(f);
    return false;
  }

  char label[pointCloudLabelBytes];
  int n = 0;
  if (fread(label, 1, pointCloudLabelBytes, f) != pointCloudLabelBytes
      || fread(&n, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "%s: file too short: %lld bytes, the header alone is %lld\n",
            path, fileBytes, pointCloudHeaderBytes);
    fclose(f);
    return false;
  }
  if (memcmp(label, pointCloudLabel, pointCloudLabelBytes) != 0) {
    fprintf(stderr, "%s: wrong label, not a point-cloud file (expected \"%s\")\n",
            path, pointCloudLabel);
    fclose(f);
    return false;
  }
  if (n <= 0) {
    fprintf(stderr, "%s: dot count is %d, must be at least 1\n", path, n);
    fclose(f);
    return false;
  }
  const long long expectedBytes = pointCloudHeaderBytes + n * pointCloudBytesPerDot;
  if (fileBytes < expectedBytes) {
    fprintf(stderr, "%s: file too short: %lld bytes, %d dots need %lld\n",
            path, fileBytes, n, expectedBytes);
    fclose(f);
    return false;
  }
  if (fileBytes > expectedBytes) {
    fprintf(stderr, "%s: count mismatch: %lld bytes, %d dots need %lld\n",
            path, fileBytes, n, expectedBytes);
    fclose(f);
    return false;
  }

  pts.resize(static_cast<size_t>(n));
  colors.resize(static_cast<size_t>(n));
  const bool ok = fread(pts.data(), sizeof(Point), pts.size(), f) == pts.size()
               && fread(colors.data(), sizeof(Color), colors.size(), f) == colors.size();
  fclose(f);
  if (!ok) {
    fprintf(stderr, "%s: read failed\n", path);
    pts.clear();
    colors.clear();
  }
  return ok;
}

#endif
