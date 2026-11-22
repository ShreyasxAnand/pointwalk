#ifndef MOVE_H
#define MOVE_H

#include <cmath>   // sinf, cosf

// Step 3: where the camera is and which way it faces, and how one key press
// changes that. Plain C++, no CUDA in here, so it can be tested on this machine.

// Everything setPose needs to place the camera.
struct Pose {
  float x, y, z;      // where the camera stands in the world, in metres
  float yaw, pitch;   // which way it faces, in radians (same meaning as in setPose)
};

// applyMove: change the pose by one small step. The viewer calls this once per picture.
//
// INPUTS (the viewer has already turned "which keys are held" into these amounts)
//   pose      the camera's pose. Passed by reference (the &), so you change the caller's pose.
//   forward   metres to walk forward this picture (negative = backward)
//   right     metres to side-step right          (negative = left)
//   up        metres to rise                     (negative = sink)
//   turn      radians to turn right              (negative = turn left)
//   look      radians to tilt up                 (negative = tilt down)
//
// ORDER: turn first, then walk, so the walk uses the direction you just turned to.
// Walking ignores pitch: W moves along the ground even while looking up or down.
// y points down, so rising makes y smaller.
inline void applyMove(Pose& pose, float forward, float right, float up, float turn, float look)
{
  pose.yaw += turn;
  pose.pitch += look;
  pose.x += forward * sinf(pose.yaw) + right * cosf(pose.yaw);
  pose.z += -1 * right * sinf(pose.yaw) + forward * cosf(pose.yaw);
  pose.y -= up;
}

#endif
