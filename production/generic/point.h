#ifndef POINTWALK_GENERIC_POINT_H
#define POINTWALK_GENERIC_POINT_H

// One dot (metres). Its frame (sensor, vehicle or world) depends on who made it.
// Colours live in a parallel list: colors[i] belongs to pts[i].
struct Point {
  float x, y, z;
};

// Written straight to disk by pointcloud.h: the layout must not change.
static_assert(sizeof(Point) == 12, "Point must be 3 floats (12 bytes) with no padding");

#endif
