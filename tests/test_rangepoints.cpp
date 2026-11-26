#include "rangepoints.h"
#include "camera.h"   // Camera, setPose, worldToSensor: for the "there and back" checks
#include "pano.h"     // panoProject: panoUnproject must be its exact reverse
#include <cstdio>
#include <cmath>
#include <vector>

// Tests for the range-image-to-dots math in src/rangepoints.h.
// Run with:  ./tests/run.sh
//
// The conventions being tested (written at the top of rangepoints.h):
//   turn   0 is straight ahead, positive is to the RIGHT
//   tilt   0 is level,          positive is DOWN
//   axes   x right, y DOWN, z ahead
// Every expected number in parts 1, 3 and 4 was worked out by hand from those.

// ---------------------------------------------------------------------------
// Helpers, shared by all the cases below.
// ---------------------------------------------------------------------------

int failures = 0;   // how many checks have failed so far

const float pi = 3.14159265f;

// Degrees to radians, because the functions expect radians.
float rad(float degrees) { return degrees * pi / 180.0f; }

// "Close enough", in metres. After sin and cos, floats are rarely exactly
// equal: cosf of 90 degrees is -0.00000004, not 0. A tenth of a millimetre is
// far tighter than anything a wrong sign or a swapped axis could pass.
bool closeTo(float a, float b) { return fabsf(a - b) < 1e-4f; }

// The same, for the checks that go through panoProject's column and row. A
// column number near 1000 is only exact to about 0.0001 of a column, and that
// small error comes back as a few hundredths of a millimetre at 60 m. One
// millimetre leaves room for it.
bool closeToMm(float a, float b) { return fabsf(a - b) < 1e-3f; }

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

// Run one (tilt, turn, distance) through panoUnproject and check the dot.
// Tilt and turn are given in degrees here, to keep the table readable.
void expectDot(const char* name, float tiltDegrees, float turnDegrees, float distance,
               float ex, float ey, float ez) {
    float x, y, z;
    panoUnproject(rad(tiltDegrees), rad(turnDegrees), distance, x, y, z);
    bool ok = closeTo(x, ex) && closeTo(y, ey) && closeTo(z, ez);
    report(name, ok);
    if (!ok) printf("      got (%f, %f, %f), expected (%f, %f, %f)\n", x, y, z, ex, ey, ez);
}

// A dot -> panoProject -> column, row, distance -> back to turn and tilt ->
// panoUnproject -> must be the dot we started with.
void expectThereAndBack(const char* name, float x, float y, float z) {
    const int W = 1024, H = 512;
    float u, v, distance;
    panoProject(x, y, z, W, H, u, v, distance);
    // Undo panoProject's last step for each: column -> turn, row -> tilt.
    float turn = u * (2 * pi) / W - pi;
    float tilt = v * pi / H - pi / 2;
    float bx, by, bz;
    panoUnproject(tilt, turn, distance, bx, by, bz);
    bool ok = closeToMm(bx, x) && closeToMm(by, y) && closeToMm(bz, z);
    report(name, ok);
    if (!ok) printf("      started at (%f, %f, %f), came back as (%f, %f, %f)\n", x, y, z, bx, by, bz);
}

// Run one dot through sensorToScene and check where it ends up.
void expectPlaced(const char* name, const SensorToScene& place,
                  float xs, float ys, float zs, float ex, float ey, float ez) {
    float x, y, z;
    sensorToScene(xs, ys, zs, place, x, y, z);
    bool ok = closeTo(x, ex) && closeTo(y, ey) && closeTo(z, ez);
    report(name, ok);
    if (!ok) printf("      got (%f, %f, %f), expected (%f, %f, %f)\n", x, y, z, ex, ey, ez);
}

// Check one dot of a finished list against the dot we expect in that box.
void expectBox(const char* name, const std::vector<Point>& points, int W, int row, int col,
               float ex, float ey, float ez) {
    const Point& p = points[(size_t)row * W + col];
    bool ok = closeTo(p.x, ex) && closeTo(p.y, ey) && closeTo(p.z, ez);
    report(name, ok);
    if (!ok) printf("      box (row %d, column %d) holds (%f, %f, %f), expected (%f, %f, %f)\n",
                    row, col, p.x, p.y, p.z, ex, ey, ez);
}

int main()
{
    // =======================================================================
    // Part 1. panoUnproject (function A): answers you can see without a formula.
    // =======================================================================

    // ---- Level, along one direction ----
    //                                       tilt   turn   distance      x      y      z
    expectDot(" 1. straight ahead",           0.0f,    0.0f, 1.0f,      0.0f,  0.0f,  1.0f);
    expectDot(" 2. directly right",           0.0f,   90.0f, 5.0f,      5.0f,  0.0f,  0.0f);
    expectDot(" 3. directly left",            0.0f,  -90.0f, 3.0f,     -3.0f,  0.0f,  0.0f);
    expectDot(" 4. directly behind",          0.0f,  180.0f, 5.0f,      0.0f,  0.0f, -5.0f);

    // ---- Level, diagonal: 4 across and 4 ahead is the root of 32 away ----
    expectDot(" 5. ahead and right",          0.0f,   45.0f, 5.656854f,  4.0f, 0.0f,  4.0f);
    expectDot(" 6. behind and left",          0.0f, -135.0f, 1.414214f, -1.0f, 0.0f, -1.0f);

    // ---- Tilted. Positive tilt is DOWN and y points DOWN, so the signs agree ----
    expectDot(" 7. straight down",           90.0f,    0.0f, 2.0f,      0.0f,  2.0f,  0.0f);
    expectDot(" 8. straight up",            -90.0f,    0.0f, 3.0f,      0.0f, -3.0f,  0.0f);
    expectDot(" 9. ahead and below",         45.0f,    0.0f, 5.656854f, 0.0f,  4.0f,  4.0f);

    // ---- Tilt and turn together ----
    // The worked example: 30 degrees down, 90 degrees right, 2 m away.
    // Down part: 2 * sin(30) = 1. Along the ground: 2 * cos(30) = 1.732051, all of it to the right.
    expectDot("10. the worked example",      30.0f,   90.0f, 2.0f,      1.732051f, 1.0f, 0.0f);
    // The bird on the pole from test_pano.cpp, backwards: 6 right, 8 ahead, 10 up.
    // 10 along the ground and 10 up is 45 degrees UP, so the tilt is -45.
    expectDot("11. ahead, right and above", -45.0f, 36.869898f, 14.142136f, 6.0f, -10.0f, 8.0f);
    // 3 right, 4 ahead (5 along the ground), 5 down: same turn, tilted the other way.
    expectDot("12. ahead, right and below",  45.0f, 36.869898f,  7.071068f, 3.0f,   5.0f, 4.0f);

    // ---- The dot must be exactly as far away as the distance that went in ----
    {
        bool ok = true;
        const float tilts[] = {-80.0f, -17.0f, 0.0f, 2.3f, 17.6f, 60.0f};
        const float turns[] = {-170.0f, -90.0f, -1.0f, 0.0f, 33.0f, 148.0f};
        for (float tilt : tilts)
            for (float turn : turns) {
                float x, y, z;
                panoUnproject(rad(tilt), rad(turn), 61.0f, x, y, z);
                float length = sqrtf(x * x + y * y + z * z);
                if (!closeTo(length, 61.0f)) {
                    ok = false;
                    printf("      tilt %f, turn %f: the dot is %f m away, expected 61\n", tilt, turn, length);
                }
            }
        report("13. the dot is as far away as the distance says (36 directions)", ok);
    }

    // =======================================================================
    // Part 2. panoUnproject is panoProject backwards. No hand-worked numbers
    // here: your own step-4 function is the judge.
    // =======================================================================

    expectThereAndBack("14. there and back: straight ahead",          0.0f,   0.0f,   8.0f);
    expectThereAndBack("15. there and back: directly right",          5.0f,   0.0f,   0.0f);
    expectThereAndBack("16. there and back: directly behind",         0.0f,   0.0f,  -5.0f);
    expectThereAndBack("17. there and back: ahead, right and above",  6.0f, -10.0f,   8.0f);
    expectThereAndBack("18. there and back: ahead, right and below",  3.0f,   5.0f,   4.0f);
    expectThereAndBack("19. there and back: behind, left and below", -7.0f,   1.5f, -12.0f);
    expectThereAndBack("20. there and back: far away",               30.8f,  18.4f, -49.3f);

    // The other way round: angles -> dot -> panoProject -> the same angles.
    // One column is 360/1024 degrees and one row is 180/512 degrees, so both
    // angles can be read straight off the column and the row.
    {
        const int W = 1024, H = 512;
        const float tilt = rad(17.6f), turn = rad(148.0f), range = 61.0f;
        float x, y, z, u, v, distance;
        panoUnproject(tilt, turn, range, x, y, z);
        panoProject(x, y, z, W, H, u, v, distance);
        float turnBack = u * (2 * pi) / W - pi;
        float tiltBack = v * pi / H - pi / 2;
        bool ok = fabsf(turnBack - turn) < 1e-5f && fabsf(tiltBack - tilt) < 1e-5f
               && closeToMm(distance, range);
        report("21. angles -> dot -> panoProject gives the same angles and distance", ok);
        if (!ok) printf("      went in: tilt %f, turn %f, distance %f\n"
                        "      came back: tilt %f, turn %f, distance %f\n",
                        tilt, turn, range, tiltBack, turnBack, distance);
    }

    // =======================================================================
    // Part 3. sensorToScene (function B).
    // =======================================================================

    // ---- Not turned, not moved: the dot stays where it is ----
    const SensorToScene still = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};
    expectPlaced("22. not turned, not moved", still, 1.0f, 2.0f, 3.0f,   1.0f, 2.0f, 3.0f);

    // ---- Moved only: the shift is added ----
    const SensorToScene moved = {{{1,0,0},{0,1,0},{0,0,1}}, {10.0f, -1.0f, 0.5f}};
    expectPlaced("23. moved only", moved, 1.0f, 2.0f, 3.0f,   11.0f, 1.0f, 3.5f);

    // ---- Turned only: the sensor faces 90 degrees to the right ----
    // What is ahead of the sensor is to the scene's right (+x).
    // What is to the sensor's right is behind in the scene (-z).
    // What is below the sensor is still below (+y).
    const SensorToScene turned = {{{0,0,1},{0,1,0},{-1,0,0}}, {0,0,0}};
    expectPlaced("24. turned right: a dot ahead",    turned, 0.0f, 0.0f, 5.0f,   5.0f, 0.0f,  0.0f);
    // This one tells R from R flipped over its diagonal: the flipped one gives z = +2.
    expectPlaced("25. turned right: a dot to the right", turned, 2.0f, 0.0f, 0.0f,   0.0f, 0.0f, -2.0f);
    expectPlaced("26. turned right: a dot below",    turned, 0.0f, 3.0f, 0.0f,   0.0f, 3.0f,  0.0f);

    // ---- Turned and moved: turn first, then shift ----
    // Shifting first and turning after would give (5, 0, -10) instead.
    const SensorToScene both = {{{0,0,1},{0,1,0},{-1,0,0}}, {10.0f, 0.0f, 0.0f}};
    expectPlaced("27. turned and moved", both, 0.0f, 0.0f, 5.0f,   15.0f, 0.0f, 0.0f);

    // ---- The reverse of worldToSensor ----
    // A camera at (2, -1.5, 15), turned and tilted. worldToSensor measures a
    // world dot from it. The placement that undoes that is: R flipped over its
    // diagonal, and the camera's position as the shift. Going there and back
    // must return the dot we started with.
    {
        Camera cam;
        setPose(cam, 2.0f, -1.5f, 15.0f, rad(40.0f), rad(-10.0f));
        SensorToScene back;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) back.R[r][c] = cam.R[c][r];
        back.t[0] = 2.0f; back.t[1] = -1.5f; back.t[2] = 15.0f;

        const Point dots[] = {{3.0f, 1.5f, 20.0f}, {0.0f, -0.5f, 8.0f}, {-6.0f, 2.0f, 11.0f}};
        bool ok = true;
        for (const Point& dot : dots) {
            float xs, ys, zs, x, y, z;
            worldToSensor(dot.x, dot.y, dot.z, cam, xs, ys, zs);
            sensorToScene(xs, ys, zs, back, x, y, z);
            if (!(closeTo(x, dot.x) && closeTo(y, dot.y) && closeTo(z, dot.z))) {
                ok = false;
                printf("      started at (%f, %f, %f), came back as (%f, %f, %f)\n",
                       dot.x, dot.y, dot.z, x, y, z);
            }
        }
        report("28. worldToSensor and then sensorToScene returns the dot", ok);
    }

    // =======================================================================
    // Part 4. rangeToPointsCPU: a whole (tiny) range image.
    //
    // 2 rows, 3 columns. Row 0 looks level, row 1 looks 45 degrees down.
    // Column 0 looks left, column 1 straight ahead, column 2 right.
    //
    //              col 0 (left)   col 1 (ahead)   col 2 (right)
    //   row 0          3              8               5
    //   row 1         -1           5.656854           0
    //
    // -1 and 0 are boxes with no return. H and W are different on purpose, so
    // mixing up rows and columns cannot pass.
    // =======================================================================

    const int H = 2, W = 3;
    const std::vector<float> ranges = {3.0f, 8.0f, 5.0f,   -1.0f, 5.656854f, 0.0f};
    const std::vector<float> tiltOfRow = {rad(0.0f), rad(45.0f)};
    const std::vector<float> turnOfColumn = {rad(-90.0f), rad(0.0f), rad(90.0f)};

    // ---- Sensor not turned, not moved ----
    {
        std::vector<Point> points((size_t)H * W, Point{0.0f, 0.0f, 0.0f});
        rangeToPointsCPU(ranges, H, W, tiltOfRow, turnOfColumn, still, points);
        expectBox("29. row 0, column 0: 3 m to the left",      points, W, 0, 0,  -3.0f, 0.0f, 0.0f);
        expectBox("30. row 0, column 1: 8 m ahead",            points, W, 0, 1,   0.0f, 0.0f, 8.0f);
        expectBox("31. row 0, column 2: 5 m to the right",     points, W, 0, 2,   5.0f, 0.0f, 0.0f);
        expectBox("32. row 1, column 1: ahead and below",      points, W, 1, 1,   0.0f, 4.0f, 4.0f);
        expectBox("33. row 1, column 0: range -1, box untouched", points, W, 1, 0,  0.0f, 0.0f, 0.0f);
        expectBox("34. row 1, column 2: range 0, box untouched",  points, W, 1, 2,  0.0f, 0.0f, 0.0f);
        report("35. the list is still H * W long", points.size() == (size_t)H * W);
    }

    // ---- Sensor moved by (10, 20, 30): every dot moves, empty boxes do not ----
    // With a shift, a box with range 0 that was wrongly treated as a return
    // would land on (10, 20, 30) and not stay at zero.
    {
        const SensorToScene shifted = {{{1,0,0},{0,1,0},{0,0,1}}, {10.0f, 20.0f, 30.0f}};
        std::vector<Point> points((size_t)H * W, Point{0.0f, 0.0f, 0.0f});
        rangeToPointsCPU(ranges, H, W, tiltOfRow, turnOfColumn, shifted, points);
        expectBox("36. moved sensor: row 0, column 0",          points, W, 0, 0,   7.0f, 20.0f, 30.0f);
        expectBox("37. moved sensor: row 1, column 1",          points, W, 1, 1,  10.0f, 24.0f, 34.0f);
        expectBox("38. moved sensor: range -1 stays untouched", points, W, 1, 0,   0.0f,  0.0f,  0.0f);
        expectBox("39. moved sensor: range 0 stays untouched",  points, W, 1, 2,   0.0f,  0.0f,  0.0f);
    }

    // ---- A range that is not a whole number keeps its decimals ----
    {
        const std::vector<float> one = {61.37f};
        std::vector<Point> points(1, Point{0.0f, 0.0f, 0.0f});
        rangeToPointsCPU(one, 1, 1, {rad(0.0f)}, {rad(0.0f)}, still, points);
        expectBox("40. 61.37 m ahead stays 61.37", points, 1, 0, 0,   0.0f, 0.0f, 61.37f);
    }

    // ---- A return nearer than 1 m is still a return ----
    {
        const std::vector<float> one = {0.4f};
        std::vector<Point> points(1, Point{0.0f, 0.0f, 0.0f});
        rangeToPointsCPU(one, 1, 1, {rad(0.0f)}, {rad(90.0f)}, still, points);
        expectBox("41. 0.4 m to the right is kept", points, 1, 0, 0,   0.4f, 0.0f, 0.0f);
    }

    printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;   // non-zero makes run.sh stop and show the failure
}
