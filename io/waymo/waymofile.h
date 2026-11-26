#ifndef WAYMOFILE_H
#define WAYMOFILE_H

// Readers for the three kinds of file that waymo/convert.py writes:
//   range/frameFFFF_laserL.bin   one range image          -> readWaymoRange
//   calibration.bin              every laser's calibration -> readWaymoCalibration
//   poses.bin                    every frame's vehicle pose -> readWaymoPoses
// The byte layout is in waymo/DATA_FORMAT.md, part 2. Plain C++, no CUDA in here.
//
// RAW ON PURPOSE. These hand back the numbers exactly as they are in the file:
// Waymo's row order, Waymo's column order, Waymo's axes, the beam table in
// Waymo's stored order, matrices as 16 doubles read row by row. Nothing is
// reordered, flipped or turned into angles here. That is the adapter's job.
//
// Every reader checks the 8 letters at the start, the sizes in the header and
// the size of the file. On any problem it prints one line to stderr saying
// which file and what is wrong, and returns false. Nothing it returned before
// the false may be used.
//
// The files are little-endian, which is the byte order of x86 and ARM
// machines, so the numbers are read straight into int, float and double.

#include <cstdio>
#include <cstring>
#include <vector>

static_assert(sizeof(int) == 4 && sizeof(long long) == 8, "the files use 4-byte int and 8-byte long long");
static_assert(sizeof(float) == 4 && sizeof(double) == 8, "the files use 4-byte float and 8-byte double");

// One range image: one laser of one frame, first return.
struct WaymoRangeImage {
  long long timestamp;         // the frame's time, in microseconds
  int frame;                   // frame number: place in time order, from 0
  int laser;                   // Waymo's laser name, 1 to 5
  int H, W, C;                 // rows, columns, numbers per box
  std::vector<float> values;   // H * W * C numbers, in the file's order:
                               //   channel c of the box at (row, col) is
                               //   values[(row * W + col) * C + c]
                               // A box with no return holds -1.
};

// One laser's calibration.
struct WaymoLaserCalibration {
  int laser;                        // Waymo's laser name, 1 to 5
  double extrinsic[16];             // 4x4 matrix, read row by row
  double beamMin, beamMax;          // smallest and largest beam angle, radians
  std::vector<double> beamValues;   // the beam angle table in Waymo's stored
                                    // order, radians. EMPTY when Waymo gives
                                    // no table for this laser.
};

// One frame's vehicle pose.
struct WaymoPose {
  long long timestamp;          // the frame's time, in microseconds
  double worldFromVehicle[16];  // 4x4 matrix, read row by row
};

// ---- helpers used only in this file ----

// Print "which file: what is wrong" and return false, so a reader can write
// "return waymoFail(...)".
inline bool waymoFail(FILE* f, const char* path, const char* what) {
  if (f) fclose(f);
  fprintf(stderr, "waymo file error: %s: %s\n", path, what);
  return false;
}

// The size of an open file in bytes. Leaves the file at its start.
inline long long waymoFileSize(FILE* f) {
  fseek(f, 0, SEEK_END);
  long long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  return size;
}

// ---- the three readers ----

// Read one range image file. Returns false (after printing why) if the file is
// missing, is not a range image file, has impossible sizes, or is not exactly
// as long as its header says.
inline bool readWaymoRange(const char* path, WaymoRangeImage& image) {
  const long long headerBytes = 36;
  char why[200];

  FILE* f = fopen(path, "rb");
  if (!f) return waymoFail(f, path, "cannot open the file");
  long long size = waymoFileSize(f);
  if (size < headerBytes) {
    snprintf(why, sizeof why, "file is %lld bytes, shorter than the %lld-byte header", size, headerBytes);
    return waymoFail(f, path, why);
  }

  char magic[8];
  bool ok = fread(magic, 1, 8, f) == 8
         && fread(&image.timestamp, sizeof(long long), 1, f) == 1
         && fread(&image.frame, sizeof(int), 1, f) == 1
         && fread(&image.laser, sizeof(int), 1, f) == 1
         && fread(&image.H, sizeof(int), 1, f) == 1
         && fread(&image.W, sizeof(int), 1, f) == 1
         && fread(&image.C, sizeof(int), 1, f) == 1;
  if (!ok) return waymoFail(f, path, "could not read the header");
  if (memcmp(magic, "PWRANGE1", 8) != 0)
    return waymoFail(f, path, "does not start with PWRANGE1, so it is not a range image file");

  // Sizes must be positive, and small enough that H * W * C cannot overflow.
  if (image.H <= 0 || image.W <= 0 || image.C <= 0 ||
      image.H > 100000 || image.W > 100000 || image.C > 64) {
    snprintf(why, sizeof why, "impossible size in the header: H %d, W %d, C %d", image.H, image.W, image.C);
    return waymoFail(f, path, why);
  }
  long long count = (long long)image.H * image.W * image.C;
  long long expected = headerBytes + count * (long long)sizeof(float);
  if (size != expected) {
    snprintf(why, sizeof why, "file is %lld bytes, but H %d x W %d x C %d needs exactly %lld",
             size, image.H, image.W, image.C, expected);
    return waymoFail(f, path, why);
  }

  image.values.resize((size_t)count);
  if (fread(image.values.data(), sizeof(float), (size_t)count, f) != (size_t)count)
    return waymoFail(f, path, "could not read all the values");
  fclose(f);
  return true;
}

// Read calibration.bin: one entry per laser, in the file's order (smallest
// laser name first). Returns false (after printing why) on any problem.
inline bool readWaymoCalibration(const char* path, std::vector<WaymoLaserCalibration>& lasers) {
  const long long headerBytes = 12;
  const long long fixedBytesPerLaser = 4 + 4 + 16 * 8 + 8 + 8;   // name, N, matrix, min, max
  char why[200];
  lasers.clear();

  FILE* f = fopen(path, "rb");
  if (!f) return waymoFail(f, path, "cannot open the file");
  long long size = waymoFileSize(f);
  if (size < headerBytes) {
    snprintf(why, sizeof why, "file is %lld bytes, shorter than the %lld-byte header", size, headerBytes);
    return waymoFail(f, path, why);
  }

  char magic[8];
  int numLasers = 0;
  if (fread(magic, 1, 8, f) != 8 || fread(&numLasers, sizeof(int), 1, f) != 1)
    return waymoFail(f, path, "could not read the header");
  if (memcmp(magic, "PWCALIB1", 8) != 0)
    return waymoFail(f, path, "does not start with PWCALIB1, so it is not a calibration file");
  if (numLasers <= 0 || numLasers > 64) {
    snprintf(why, sizeof why, "impossible number of lasers in the header: %d", numLasers);
    return waymoFail(f, path, why);
  }

  long long used = headerBytes;   // bytes of the file accounted for so far
  for (int i = 0; i < numLasers; i++) {
    if (size - used < fixedBytesPerLaser) {
      snprintf(why, sizeof why, "file ends inside laser entry %d of %d", i + 1, numLasers);
      return waymoFail(f, path, why);
    }
    WaymoLaserCalibration cal;
    int numBeams = 0;
    bool ok = fread(&cal.laser, sizeof(int), 1, f) == 1
           && fread(&numBeams, sizeof(int), 1, f) == 1
           && fread(cal.extrinsic, sizeof(double), 16, f) == 16
           && fread(&cal.beamMin, sizeof(double), 1, f) == 1
           && fread(&cal.beamMax, sizeof(double), 1, f) == 1;
    if (!ok) return waymoFail(f, path, "could not read a laser entry");
    used += fixedBytesPerLaser;

    // The table must fit in what is left of the file.
    if (numBeams < 0 || (long long)numBeams * 8 > size - used) {
      snprintf(why, sizeof why, "laser %d says it has %d beam angles, which does not fit in the file", cal.laser, numBeams);
      return waymoFail(f, path, why);
    }
    cal.beamValues.resize((size_t)numBeams);
    if (numBeams > 0 && fread(cal.beamValues.data(), sizeof(double), (size_t)numBeams, f) != (size_t)numBeams)
      return waymoFail(f, path, "could not read a beam angle table");
    used += (long long)numBeams * 8;
    lasers.push_back(cal);
  }

  if (used != size) {
    snprintf(why, sizeof why, "file is %lld bytes, but its %d laser entries take exactly %lld", size, numLasers, used);
    lasers.clear();
    return waymoFail(f, path, why);
  }
  fclose(f);
  return true;
}

// Read poses.bin: one entry per frame, in time order, so poses[k] belongs to
// frame number k. Returns false (after printing why) on any problem.
inline bool readWaymoPoses(const char* path, std::vector<WaymoPose>& poses) {
  const long long headerBytes = 12;
  const long long bytesPerFrame = 8 + 16 * 8;   // timestamp, matrix
  char why[200];
  poses.clear();

  FILE* f = fopen(path, "rb");
  if (!f) return waymoFail(f, path, "cannot open the file");
  long long size = waymoFileSize(f);
  if (size < headerBytes) {
    snprintf(why, sizeof why, "file is %lld bytes, shorter than the %lld-byte header", size, headerBytes);
    return waymoFail(f, path, why);
  }

  char magic[8];
  int numFrames = 0;
  if (fread(magic, 1, 8, f) != 8 || fread(&numFrames, sizeof(int), 1, f) != 1)
    return waymoFail(f, path, "could not read the header");
  if (memcmp(magic, "PWPOSES1", 8) != 0)
    return waymoFail(f, path, "does not start with PWPOSES1, so it is not a poses file");
  if (numFrames <= 0) {
    snprintf(why, sizeof why, "impossible number of frames in the header: %d", numFrames);
    return waymoFail(f, path, why);
  }
  long long expected = headerBytes + (long long)numFrames * bytesPerFrame;
  if (size != expected) {
    snprintf(why, sizeof why, "file is %lld bytes, but %d frames need exactly %lld", size, numFrames, expected);
    return waymoFail(f, path, why);
  }

  poses.resize((size_t)numFrames);
  for (int i = 0; i < numFrames; i++) {
    bool ok = fread(&poses[i].timestamp, sizeof(long long), 1, f) == 1
           && fread(poses[i].worldFromVehicle, sizeof(double), 16, f) == 16;
    if (!ok) {
      poses.clear();
      return waymoFail(f, path, "could not read a pose");
    }
  }
  fclose(f);
  return true;
}

#endif
