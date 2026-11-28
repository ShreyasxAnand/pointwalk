#ifndef POINTWALK_GENERIC_COLOR_H
#define POINTWALK_GENERIC_COLOR_H

// One colour: red, green, blue, each 0 to 255.
struct Color {
  unsigned char r, g, b;
};

// Written straight to disk by pointcloud.h: the layout must not change.
static_assert(sizeof(Color) == 3, "Color must be 3 bytes with no padding");

#endif
