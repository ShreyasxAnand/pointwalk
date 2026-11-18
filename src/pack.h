#ifndef PACK_H
#define PACK_H

// Step 2: glue a distance and a dot number into one 64-slot number, and split
// them apart again. Plain C++, so it can be tested on this machine with g++.

#include <cstring>   // memcpy

// These functions are needed in two places: on the CPU (the test and the CPU
// renderer) and on the GPU (the kernel). Writing HD in front of a function
// tells the CUDA compiler "build this for both". Under plain g++, HD means nothing.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// The two kinds of whole number used here:
//   unsigned int   32 slots. Holds a dot number, or the raw slots of a float.
//   Packed         64 slots. Holds a glued (distance, dot number) pair.
// (Packed is "unsigned long long" because that is the exact type the GPU's
// atomicMin accepts.)
typedef unsigned long long Packed;

// THE LAYOUT
//
//     [ distance: left 32 slots ][ dot number: right 32 slots ]
//
//   Same idea as "distance * 1000 + dot number", with 32 slots instead of
//   3 digits. The distance is on the left, so the smaller Packed number is
//   always the closer dot. This only holds for positive distances.
//
// THE FOUR MOVES
//   slide 32 slots left        x << 32     (like * 1000)
//   drop into the empty half   a | b       (like + dot number)
//   slide 32 slots right       x >> 32     (like / 1000, remainder thrown away)
//   keep the right 32 slots    (unsigned int)x   (like keeping the last 3 digits)
//
// TWO FACTS YOU NEED
//   1. The distance is a float, but it goes in as its 32 raw slots, NOT rounded
//      to whole metres. This is the step 0 idea: for positive floats, the raw
//      slots compare the same way the floats do. To copy a float's slots into
//      a whole number, and back (works on both CPU and GPU):
//          unsigned int bits;  memcpy(&bits, &depth, 4);     // float -> slots
//          float depth;        memcpy(&depth, &bits, 4);     // slots -> float
//   2. Make the number 64 slots wide BEFORE sliding it left: write (Packed)bits << 32.
//      A 32-slot number has no room to slide 32 places.

// Glue: distance on the left, dot number on the right.
//   Example with the digits picture: 3 m and dot 250  ->  3250
HD inline Packed packDepthIndex(float depth, unsigned int index) {
    unsigned int bits;
    memcpy(&bits, &depth, 4);               // the float's 32 raw slots, unchanged
    return ((Packed)bits << 32) | index;    // widen, slide left, drop the dot number into the right half
}

// Split: get the distance back out of a glued number.
//   Example with the digits picture: 3250  ->  3 m
HD inline float unpackDepth(Packed packed) {
    unsigned int bits = (unsigned int)(packed >> 32);   // slide right: the left half is now on its own
    float depth;
    memcpy(&depth, &bits, 4);                           // those 32 slots are a float again
    return depth;
}

// Split: get the dot number back out of a glued number.
//   Example with the digits picture: 3250  ->  dot 250
HD inline unsigned int unpackIndex(Packed packed) {
    return (unsigned int)packed;            // keep only the right 32 slots
}

#endif
