#include "camera.h"
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// Helpers, shared by all the cases below.
// ---------------------------------------------------------------------------

int failures = 0;   // how many checks have failed so far

// Degrees to radians, because setPose expects radians.
float rad(float degrees) { return degrees * 3.14159265f / 180.0f; }

// "Close enough". After sin and cos, floats are rarely exactly equal:
// cosf of 90 degrees is -0.00000004, not 0. So we allow a tiny difference.
bool closeTo(float a, float b) { return fabsf(a - b) < 1e-5f; }

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

// Check cam.t against the shift we expect.
void expectT(const char* name, const Camera& cam, float tx, float ty, float tz) {
    bool ok = closeTo(cam.t[0], tx) && closeTo(cam.t[1], ty) && closeTo(cam.t[2], tz);
    report(name, ok);
    if (!ok) printf("      got t = (%f, %f, %f), expected (%f, %f, %f)\n",
                    cam.t[0], cam.t[1], cam.t[2], tx, ty, tz);
}

// Check where the camera sees a world point (x, y, z).
// The three lines for xc, yc, zc are the same ones the kernel uses.
void expectSeen(const char* name, const Camera& cam,
                float x, float y, float z, float ex, float ey, float ez) {
    float xc = cam.R[0][0]*x + cam.R[0][1]*y + cam.R[0][2]*z + cam.t[0];
    float yc = cam.R[1][0]*x + cam.R[1][1]*y + cam.R[1][2]*z + cam.t[1];
    float zc = cam.R[2][0]*x + cam.R[2][1]*y + cam.R[2][2]*z + cam.t[2];
    bool ok = closeTo(xc, ex) && closeTo(yc, ey) && closeTo(zc, ez);
    report(name, ok);
    if (!ok) printf("      got (%f, %f, %f), expected (%f, %f, %f)\n", xc, yc, zc, ex, ey, ez);
}

int main(int argc, char const *argv[])
{
    // TEST CASE 1: yaw 0, pitch 0, C = (0,0,0)   ->  R = identity, t = (0,0,0)
    Camera cam;
    setPose(cam, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    printf("Test Case 1:\n");
    printf("R = [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n", cam.R[0][0], cam.R[0][1], cam.R[0][2],
        cam.R[1][0], cam.R[1][1], cam.R[1][2], cam.R[2][0], cam.R[2][1], cam.R[2][2]);
    printf("t = [%.6f %.6f %.6f]\n", cam.t[0], cam.t[1], cam.t[2]);

    // Same case, checked automatically. Here we want EXACT equality, not "close":
    // step 1's first check needs this pose to give the identical image.
    bool identity = true;
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (cam.R[row][col] != (row == col ? 1.0f : 0.0f)) identity = false;
    report("1. no movement: R is exactly the identity", identity);
    report("1. no movement: t is exactly zero",
           cam.t[0] == 0.0f && cam.t[1] == 0.0f && cam.t[2] == 0.0f);

    // ---- Moving without turning: t = -C ----

    // 2. Step back 1 m. Everything is 1 m farther ahead.
    setPose(cam, 0, 0, -1, 0, 0);
    expectT   ("2. back 1 m: t = (0,0,1)", cam, 0, 0, 1);
    expectSeen("2. back 1 m: point 5 m ahead is now 6 m ahead", cam, 0, 0, 5,  0, 0, 6);

    // 3. Move 2 m right and 3 m forward.
    setPose(cam, 2, 0, 3, 0, 0);
    expectT   ("3. right 2, forward 3: t = (-2,0,-3)", cam, -2, 0, -3);
    expectSeen("3. right 2, forward 3: (2,0,10) is 7 m straight ahead", cam, 2, 0, 10,  0, 0, 7);
    expectSeen("3. right 2, forward 3: (0,0,5) is 2 m left, 2 m ahead", cam, 0, 0, 5,  -2, 0, 2);

    // ---- Yaw: turning right (+) or left (-) ----

    // 4. Turn right 90 degrees. What was ahead is now on the left.
    setPose(cam, 0, 0, 0, rad(90), 0);
    expectSeen("4. yaw +90: point ahead is now on the left", cam, 0, 0, 5,  -5, 0, 0);
    expectSeen("4. yaw +90: point on the right is now ahead", cam, 5, 0, 0,  0, 0, 5);

    // 5. Turn left 90 degrees. What was ahead is now on the right.
    setPose(cam, 0, 0, 0, rad(-90), 0);
    expectSeen("5. yaw -90: point ahead is now on the right", cam, 0, 0, 5,  5, 0, 0);

    // 6. Turn around. What was ahead is now behind (negative z).
    setPose(cam, 0, 0, 0, rad(180), 0);
    expectSeen("6. yaw 180: point ahead is now behind", cam, 0, 0, 5,  0, 0, -5);

    // 7. Turn right 45 degrees. A point 1.414 m ahead ends up 1 m left and 1 m ahead.
    setPose(cam, 0, 0, 0, rad(45), 0);
    expectSeen("7. yaw +45: point ahead is ahead-left", cam, 0, 0, sqrtf(2.0f),  -1, 0, 1);

    // 8. Yaw never changes height: y passes straight through.
    setPose(cam, 0, 0, 0, rad(30), 0);
    expectSeen("8. yaw +30: a point 3 m above the camera stays 3 m above", cam, 0, -3, 0,  0, -3, 0);

    // ---- Pitch: looking up (+) or down (-). Remember y points DOWN. ----

    // 9. Look straight up. A point directly overhead (y = -5) is now straight ahead,
    //    and what was ahead has dropped to the bottom of the view (positive y).
    setPose(cam, 0, 0, 0, 0, rad(90));
    expectSeen("9. pitch +90: point overhead is now ahead", cam, 0, -5, 0,  0, 0, 5);
    expectSeen("9. pitch +90: point ahead is now below", cam, 0, 0, 5,  0, 5, 0);

    // 10. Look straight down. The floor point under the camera (y = +5) is now ahead.
    setPose(cam, 0, 0, 0, 0, rad(-90));
    expectSeen("10. pitch -90: point underneath is now ahead", cam, 0, 5, 0,  0, 0, 5);

    // ---- Everything together ----

    // 11. Stand 10 m to the right of the origin and turn left to face it.
    //     The origin should be 10 m straight ahead.
    setPose(cam, 10, 0, 0, rad(-90), 0);
    expectSeen("11. at (10,0,0) facing left: origin is 10 m ahead", cam, 0, 0, 0,  0, 0, 10);

    // 12. Turn right 90, then look straight up. A point overhead is straight ahead.
    setPose(cam, 0, 0, 0, rad(90), rad(90));
    expectSeen("12. yaw +90, pitch +90: point overhead is ahead", cam, 0, -5, 0,  0, 0, 5);

    // 13. For ANY pose, the camera's own position must come out as (0,0,0):
    //     the camera is always at the origin of its own view.
    setPose(cam, 2, -1, 3, 0.7f, -0.4f);
    expectSeen("13. any pose: the camera sees its own position at (0,0,0)", cam, 2, -1, 3,  0, 0, 0);

    // 14. For ANY pose, R must be a pure rotation: every row has length 1, and
    //     the rows are at right angles to each other. Otherwise R would stretch
    //     or squash the scene. (Row i dotted with row j is 1 if i == j, else 0.)
    bool rotation = true;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float dot = cam.R[i][0]*cam.R[j][0] + cam.R[i][1]*cam.R[j][1] + cam.R[i][2]*cam.R[j][2];
            if (!closeTo(dot, i == j ? 1.0f : 0.0f)) rotation = false;
        }
    report("14. any pose: R is a pure rotation", rotation);

    printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;   // non-zero makes run.sh stop and show the failure
}
