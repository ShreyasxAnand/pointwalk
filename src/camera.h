#ifndef CAMERA_H
#define CAMERA_H
#include <cmath>

// Shared by the GPU programs and the CPU tests. Plain C++, no CUDA in here.

// Used by project.cu and the tests. bench.cu still has its own copy of this struct.
struct Camera {
  float R[3][3];         // rotation: world -> camera
  float t[3];            // shift:    world -> camera
  float fx, fy, cx, cy;  // focal lengths and image center, in pixels
  int W, H;              // image size, in pixels
};

// setPose: place the camera somewhere and point it in a direction.
//
// WHAT IT DOES
//   Fills in cam.R (9 numbers) and cam.t (3 numbers). It does not touch the
//   lens numbers (fx, fy, cx, cy, W, H).
//   The kernel then uses them on every point:
//       camera point = R * world point + t
//   which re-measures the point from the camera's point of view.
//
// AXES (same in world space and camera space)
//   x = right,  y = down,  z = forward
//
// INPUTS
//   cam          the camera to update. It is passed by reference (the &), so
//                the function changes the caller's camera, not a copy.
//   Cx, Cy, Cz   where the camera stands in the world, in metres.
//   yaw          turn right (+) or left (-), in RADIANS. 0 = facing +z.
//   pitch        look up (+) or down (-), in RADIANS. 0 = level.
//   Degrees to radians: radians = degrees * 3.14159265f / 180.0f
//
// STEP 1: four helper values
//   sy = sinf(yaw)     cy = cosf(yaw)
//   sp = sinf(pitch)   cp = cosf(pitch)
//
// STEP 2: R. Each ROW is one of the camera's own directions, written in
//   world coordinates.
//
//                   column 0    column 1    column 2
//   row 0 (right)     cy           0          -sy
//   row 1 (down)      sy*sp        cp          cy*sp
//   row 2 (forward)   sy*cp       -sp          cy*cp
//
//   cam.R[row][column], so for example cam.R[0][2] = -sy.
//   With yaw = 0 and pitch = 0 this is the identity: 1 on the diagonal, 0 elsewhere.
//
// STEP 3: t = -R * C. One number per row: minus (that row dotted with C).
//   cam.t[0] = -(R[0][0]*Cx + R[0][1]*Cy + R[0][2]*Cz)
//   cam.t[1] = -(R[1][0]*Cx + R[1][1]*Cy + R[1][2]*Cz)
//   cam.t[2] = -(R[2][0]*Cx + R[2][1]*Cy + R[2][2]*Cz)
//   With no rotation this is just t = -C.
//
// WHY t HAS A MINUS SIGN
//   The camera never moves in the code. Stepping the camera back 1 m is done
//   by pushing every point 1 m forward instead.
//
// WORKED EXAMPLES (use these in the test)
//   yaw 0, pitch 0, C = (0,0,0)   ->  R = identity, t = (0,0,0)
//   yaw 0, pitch 0, C = (0,0,-1)  ->  t = (0,0,1); world (0,0,5) is seen at (0,0,6)
//   yaw 90 degrees, C = (0,0,0)   ->  world (0,0,5) is seen at (-5,0,0)
//                                     world (5,0,0) is seen at (0,0,5)
//
// NOTE
//   cosf of 90 degrees gives a tiny number like -0.00000004, not exactly 0.
//   In tests, compare with a small tolerance instead of ==.
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

#endif
