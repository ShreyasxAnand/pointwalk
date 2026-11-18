#include "pack.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int failures = 0;   // how many checks have failed so far

// Print PASS or FAIL for one check, and count the failures.
void report(const char* name, bool ok) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

int main() {
    // ---- Glue, then split: you must get back exactly what went in ----
    // Exact equality (==) is right here. Nothing is calculated, the slots are only moved.
    Packed p = packDepthIndex(3.0f, 250);
    report("1. 3 m and dot 250: distance comes back as 3", unpackDepth(p) == 3.0f);
    report("1. 3 m and dot 250: dot number comes back as 250", unpackIndex(p) == 250);

    // A distance that is not a whole number must survive untouched too.
    p = packDepthIndex(7.123456f, 19);
    report("2. 7.123456 m and dot 19: distance comes back exactly", unpackDepth(p) == 7.123456f);
    report("2. 7.123456 m and dot 19: dot number comes back as 19", unpackIndex(p) == 19);

    // The smallest and the largest dot numbers that fit in 32 slots.
    p = packDepthIndex(5.0f, 0);
    report("3. dot number 0 comes back", unpackIndex(p) == 0 && unpackDepth(p) == 5.0f);
    p = packDepthIndex(5.0f, 4294967295u);
    report("3. largest dot number comes back, and does not leak into the distance",
           unpackIndex(p) == 4294967295u && unpackDepth(p) == 5.0f);

    // ---- The reason for the whole trick: smaller glued number = closer dot ----
    report("4. 2 m beats 3 m, even though its dot number is bigger",
           packDepthIndex(2.0f, 7) < packDepthIndex(3.0f, 2));
    report("5. closer wins even by the tiniest amount, with the worst dot numbers",
           packDepthIndex(5.0f, 4294967295u) < packDepthIndex(nextafterf(5.0f, 10.0f), 0));
    report("6. same distance: the lower dot number wins",
           packDepthIndex(2.0f, 3) < packDepthIndex(2.0f, 7));
    report("7. any real dot beats an empty note (1e30 = really far)",
           packDepthIndex(30.0f, 4294967295u) < packDepthIndex(1e30f, 0));

    // ---- Lots of random pairs: the round trip and the ordering must always hold ----
    srand(1);
    bool roundTrip = true, ordering = true;
    for (int k = 0; k < 1000000; k++) {
        float da = 0.1f + 30.0f * ((float)rand() / (float)RAND_MAX);   // distances the renderer can produce
        float db = 0.1f + 30.0f * ((float)rand() / (float)RAND_MAX);
        unsigned int ia = (unsigned int)rand(), ib = (unsigned int)rand();
        Packed a = packDepthIndex(da, ia), b = packDepthIndex(db, ib);
        if (unpackDepth(a) != da || unpackIndex(a) != ia) roundTrip = false;
        if (da < db && !(a < b)) ordering = false;    // closer must mean smaller
        if (da > db && !(a > b)) ordering = false;
    }
    report("8. one million random pairs: split gives back what was glued", roundTrip);
    report("9. one million random pairs: closer always means smaller", ordering);

    printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
