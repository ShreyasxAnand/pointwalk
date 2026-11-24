#ifndef CAMERA_H
#define CAMERA_H
#include <cmath>

// Shared by the GPU programs and the CPU tests. Plain C++, no CUDA in here.

// Same struct as in project.cu and bench.cu.
struct Camera {
  float R[3][3];         // rotation: world -> camera
  float t[3];            // shift:    world -> camera
  float fx, fy, cx, cy;  // focal lengths and image center, in pixels
  int W, H;              // image size, in pixels
};

inline void setPose(Camera& cam, float Cx, float Cy, float Cz, float yaw, float pitch)
{
    float sinYaw = sinf(yaw);
    float cosYaw = cosf(yaw);
    float sinPitch = sinf(pitch);
    float cosPitch = cosf(pitch);
    cam.R[0][0] = cosYaw;
    cam.R[0][1] = 0.0f;
    cam.R[0][2] = -sinYaw;
    cam.R[1][0] = sinYaw * sinPitch;
    cam.R[1][1] = cosPitch;
    cam.R[1][2] = cosYaw * sinPitch;
    cam.R[2][0] = sinYaw * cosPitch;
    cam.R[2][1] = -sinPitch;
    cam.R[2][2] = cosYaw * cosPitch;
    cam.t[0] = -(cam.R[0][0]*Cx + cam.R[0][1]*Cy + cam.R[0][2]*Cz);
    cam.t[1] = -(cam.R[1][0]*Cx + cam.R[1][1]*Cy + cam.R[1][2]*Cz);
    cam.t[2] = -(cam.R[2][0]*Cx + cam.R[2][1]*Cy + cam.R[2][2]*Cz);
}

// HD in front of a function tells the CUDA compiler "build this for the CPU and
// the GPU". Under plain g++, HD means nothing. (Same macro as in pack.h.)
#ifndef HD
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

// worldToSensor: re-measure one dot from the camera's (or sensor's) point of view.
//
// INPUTS
//   wx, wy, wz   the dot in world coordinates, in metres
//   cam          the camera. Only cam.R and cam.t are used; setPose filled them in.
//
// OUTPUTS (passed by reference, so you fill in the caller's variables)
//   xs           how far the dot is to the sensor's right
//   ys           how far the dot is BELOW the sensor (y points down)
//   zs           how far the dot is ahead of the sensor (negative = behind)
//
// It does no skipping and no dividing: a dot behind the sensor still gets its
// three numbers. What to do with them is the caller's business.
//
// WORKED EXAMPLES (use these in the test)
//   sensor at (0, 0, 15), not turned:   world (3, 1.5, 20)   ->  (3, 1.5, 5)
//                                       world (0, -0.5, 8)   ->  (0, -0.5, -7)
//   sensor at (0, 0, 0), yaw 90 degrees to the right:
//                                       world (0, 0, 5)      ->  (-5, 0, 0)
HD inline void worldToSensor(float wx, float wy, float wz, const Camera& cam,
                             float& xs, float& ys, float& zs)
{
   xs = cam.R[0][0] * wx + cam.R[0][1] * wy + cam.R[0][2] * wz + cam.t[0];
   ys = cam.R[1][0] * wx + cam.R[1][1] * wy + cam.R[1][2] * wz + cam.t[1];
   zs = cam.R[2][0] * wx + cam.R[2][1] * wy + cam.R[2][2] * wz + cam.t[2];
}

#endif
