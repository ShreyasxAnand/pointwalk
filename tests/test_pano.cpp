#include "pano.h"
#include <cstdio>
#include <cmath>
#include <vector>

// Tests for the wrap-around picture math in src/pano.h.
// Run with:  ./tests/run.sh
//
// Every expected number below was worked out by hand from the layout written
// at the top of pano.h. Most cases use a picture 1024 boxes wide and 512 tall:
//   a full turn (360 degrees) is 1024 columns, so 90 degrees is 256 columns
//   straight up to straight down (180 degrees) is 512 rows, so 90 degrees is 256 rows
//   straight ahead and level is the middle: column 512, row 256

// ---------------------------------------------------------------------------
// Helpers, shared by all the cases below.
// ---------------------------------------------------------------------------

int failures = 0;   // how many checks have failed so far

// "Close enough". A column can be as large as the picture is wide, and a float
// near 1000 is only exact to about 0.0001, so a tighter limit would fail for
// no real reason. 0.001 of a box is far smaller than anything you could see.
bool closeTo(float a, float b) { return fabsf(a - b) < 1e-3f; }

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

// Run one dot through panoProject and check the column, the row and the distance.
void expectPano(const char* name, float x, float y, float z, int W, int H,
                float expectedU, float expectedV, float expectedDistance) {
    float u, v, distance;
    panoProject(x, y, z, W, H, u, v, distance);
    bool ok = closeTo(u, expectedU) && closeTo(v, expectedV) && closeTo(distance, expectedDistance);
    report(name, ok);
    if (!ok) printf("      got column %f, row %f, distance %f\n"
                    "      expected column %f, row %f, distance %f\n",
                    u, v, distance, expectedU, expectedV, expectedDistance);
}

// ---------------------------------------------------------------------------
// Helpers for panoDepthCPU (the whole 360 depth picture).
// ---------------------------------------------------------------------------

// Draw the dots from the given sensor into a fresh W x H picture.
std::vector<float> drawPano(const std::vector<Point>& pts, const Camera& cam, int W, int H) {
    std::vector<float> depth((size_t)W * H, 1e30f);   // every box starts "really far"
    panoDepthCPU(pts, cam, W, H, depth);
    return depth;
}

// How many boxes have a dot in them? Also hands back the position of the last one found.
int countFilled(const std::vector<float>& depth, int& lastFilled) {
    int filled = 0;
    lastFilled = -1;
    for (int k = 0; k < (int)depth.size(); k++)
        if (depth[k] < 1e29f) { filled++; lastFilled = k; }
    return filled;
}

// One dot in, exactly one box filled: check which box and what it says.
// Pass -1 for expectedCol to skip the column check (straight up and straight down).
void expectOneBox(const char* name, Point dot, const Camera& cam, int W, int H,
                  int expectedCol, int expectedRow, float expectedDistance) {
    std::vector<float> depth = drawPano({dot}, cam, W, H);
    int at;
    int filled = countFilled(depth, at);
    int col = at % W, row = at / W;
    bool ok = filled == 1
           && (expectedCol < 0 || col == expectedCol)
           && row == expectedRow
           && closeTo(depth[at], expectedDistance);
    report(name, ok);
    if (!ok) {
        if (filled == 1) printf("      got column %d, row %d, distance %f\n", col, row, depth[at]);
        else             printf("      got %d boxes filled, expected exactly 1\n", filled);
        printf("      expected column %d, row %d, distance %f\n", expectedCol, expectedRow, expectedDistance);
    }
}

int main()
{
    const int W = 1024, H = 512;

    // ---- Straight along one direction: the answers you can see without any math ----
    //                                        x      y      z            column  row   distance
    expectPano(" 1. straight ahead",         0.0f,  0.0f,  8.0f, W, H,  512.0f, 256.0f, 8.0f);
    expectPano(" 2. directly right",         5.0f,  0.0f,  0.0f, W, H,  768.0f, 256.0f, 5.0f);
    expectPano(" 3. directly left",         -3.0f,  0.0f,  0.0f, W, H,  256.0f, 256.0f, 3.0f);

    // ---- Diagonals on the ground: 45 degrees is 128 columns ----
    // 4 across and 4 ahead: the distance is the root of 16 + 16 = 32.
    expectPano(" 4. ahead and right",        4.0f,  0.0f,  4.0f, W, H,  640.0f, 256.0f, 5.656854f);
    expectPano(" 5. ahead and left",        -2.0f,  0.0f,  2.0f, W, H,  384.0f, 256.0f, 2.828427f);
    // Behind you. Plain atan (dividing first) gets these two wrong; atan2f gets them right.
    expectPano(" 6. behind and left",       -1.0f,  0.0f, -1.0f, W, H,  128.0f, 256.0f, 1.414214f);
    expectPano(" 7. behind and right",       1.0f,  0.0f, -1.0f, W, H,  896.0f, 256.0f, 1.414214f);

    // ---- Tilted. y points DOWN, so a positive y is below you: 45 degrees is 128 rows ----
    expectPano(" 8. ahead and below",        0.0f,  4.0f,  4.0f, W, H,  512.0f, 384.0f, 5.656854f);
    // The bird on the pole: 6 right, 8 ahead (10 along the ground), 10 up.
    // The turn is atan2f(6, 8) = 36.8699 degrees = 104.8744 columns right of the middle.
    expectPano(" 9. ahead, right and above", 6.0f, -10.0f, 8.0f, W, H,  616.8744f, 128.0f, 14.142136f);
    // 3 right, 4 ahead (5 along the ground), 5 down. Same turn as case 9, tilted the other way.
    expectPano("10. ahead, right and below", 3.0f,  5.0f,  4.0f, W, H,  616.8744f, 384.0f, 7.071068f);

    // ---- A different picture size: the function must use W and H, not fixed numbers ----
    // 360 by 180 is one column and one row per degree.
    expectPano("11. directly right, 360 by 180", 5.0f, 0.0f, 0.0f, 360, 180, 270.0f, 90.0f, 5.0f);

    // ---- The edges of the picture ----
    float u, v, distance;

    // Straight up and straight down. The column is not checked: when you look
    // straight up, every turn points the same way, so any column is as good as another.
    panoProject(0.0f, -3.0f, 0.0f, W, H, u, v, distance);
    report("12. straight up is the top row", closeTo(v, 0.0f) && closeTo(distance, 3.0f));
    panoProject(0.0f, 2.0f, 0.0f, W, H, u, v, distance);
    report("13. straight down is the bottom edge", closeTo(v, 512.0f) && closeTo(distance, 2.0f));

    // Directly behind. This is where the picture wraps: the left edge (column 0)
    // and the right edge (column 1024) are the same direction, so either is right.
    panoProject(0.0f, 0.0f, -5.0f, W, H, u, v, distance);
    report("14. directly behind is the wrap edge",
           (closeTo(u, 0.0f) || closeTo(u, 1024.0f)) && closeTo(v, 256.0f) && closeTo(distance, 5.0f));

    // =======================================================================
    // panoDepthCPU: the whole picture. Each case draws one or two dots and
    // looks at which boxes got filled. One column or row is 360/1024 = 0.3516
    // degrees, so degrees * 2.8444 = boxes. The box is the whole-number part.
    // =======================================================================

    // The sensor stands in the middle of the floor, not turned (same as pano.cu).
    Camera sensor;
    setPose(sensor, 0.0f, 0.0f, 15.0f, 0.0f, 0.0f);

    // 15. No dots at all: every box must still say "really far".
    {
        int at;
        report("15. no dots: no box is filled", countFilled(drawPano({}, sensor, W, H), at) == 0);
    }

    // 16. World (4, 1.5, 21) is (4, 1.5, 6) from the sensor: right, below, ahead.
    //     turn = atan2(4, 6) = 33.690 degrees -> 95.83 columns right of 512 -> u = 607.83
    //     ground length = root(16 + 36) = 7.2111
    //     tilt = atan2(1.5, 7.2111) = 11.751 degrees -> 33.42 rows below 256 -> v = 289.42
    //     distance = root(16 + 2.25 + 36) = 7.365460
    expectOneBox("16. ahead, right and below", {4.0f, 1.5f, 21.0f}, sensor, W, H,  607, 289, 7.365460f);

    // 17. World (1, -0.5, 8) is (1, -0.5, -7) from the sensor: BEHIND, a little right, a little above.
    //     The flat camera would skip this dot. The 360 picture must not.
    //     turn = atan2(1, -7) = 171.870 degrees -> 488.87 columns right of 512 -> u = 1000.87
    //     ground length = root(1 + 49) = 7.0711
    //     tilt = atan2(-0.5, 7.0711) = -4.045 degrees -> 11.50 rows above 256 -> v = 244.50
    //     distance = root(1 + 0.25 + 49) = 7.088723
    expectOneBox("17. behind, right and above", {1.0f, -0.5f, 8.0f}, sensor, W, H,  1000, 244, 7.088723f);

    // 18. Straight down: the floor dot right under the sensor. v = 512 exactly,
    //     and there is no row 512, so it belongs in the last row, 511.
    //     The column is not checked (straight down, every turn points the same way).
    expectOneBox("18. straight down goes in the last row", {0.0f, 1.5f, 15.0f}, sensor, W, H,  -1, 511, 1.5f);

    // 19. Straight up: v = 0, the top row.
    expectOneBox("19. straight up goes in the top row", {0.0f, -4.0f, 15.0f}, sensor, W, H,  -1, 0, 4.0f);

    // 20. Two dots in the same direction, 3 m and 10 m ahead: one box, and it says 3.
    //     Tried in both orders, because the answer must not depend on which dot comes first.
    {
        int at;
        std::vector<float> nearFirst = drawPano({{0.0f, 0.0f, 18.0f}, {0.0f, 0.0f, 25.0f}}, sensor, W, H);
        bool ok = countFilled(nearFirst, at) == 1 && closeTo(nearFirst[at], 3.0f);
        std::vector<float> farFirst  = drawPano({{0.0f, 0.0f, 25.0f}, {0.0f, 0.0f, 18.0f}}, sensor, W, H);
        ok = ok && countFilled(farFirst, at) == 1 && closeTo(farFirst[at], 3.0f);
        report("20. two dots in one direction: the nearer one wins, in either order", ok);
    }

    // 21. A turned sensor: at the origin, turned right 90 degrees. Now world +x is
    //     ahead and world -z is on the right, so world (5, 1, -1) is (1, 1, 5) from the sensor.
    //     turn = atan2(1, 5) = 11.310 degrees -> 32.17 columns right of 512 -> u = 544.17
    //     ground length = root(1 + 25) = 5.0990
    //     tilt = atan2(1, 5.0990) = 11.096 degrees -> 31.56 rows below 256 -> v = 287.56
    //     distance = root(1 + 1 + 25) = 5.196152
    Camera turned;
    setPose(turned, 0.0f, 0.0f, 0.0f, 3.14159265f / 2.0f, 0.0f);
    expectOneBox("21. turned sensor: R is used, not just the position", {5.0f, 1.0f, -1.0f}, turned, W, H,  544, 287, 5.196152f);

    // 22. A different picture size, 360 by 180 (one box per degree), same dot as case 16:
    //     u = 180 + 33.690 = 213.69, v = 90 + 11.751 = 101.75.
    expectOneBox("22. 360 by 180 picture", {4.0f, 1.5f, 21.0f}, sensor, 360, 180,  213, 101, 7.365460f);

    // 23. Too close to be real: anything nearer than 0.1 m is skipped, the same
    //     rule the flat camera uses (zc < 0.1f). A dot exactly on the sensor has
    //     no direction at all, and at distance 0 it would win its box forever.
    {
        int at;
        report("23. a dot exactly on the sensor is skipped",
               countFilled(drawPano({{0.0f, 0.0f, 15.0f}}, sensor, W, H), at) == 0);
        report("23. a dot 0.05 m away is skipped",
               countFilled(drawPano({{0.0f, 0.0f, 15.05f}}, sensor, W, H), at) == 0);
        std::vector<float> kept = drawPano({{0.0f, 0.0f, 15.2f}}, sensor, W, H);
        report("23. a dot 0.2 m away is kept",
               countFilled(kept, at) == 1 && closeTo(kept[at], 0.2f));
    }

    printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;   // non-zero makes run.sh stop and show the failure
}
