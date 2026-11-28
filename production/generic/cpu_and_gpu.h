#ifndef POINTWALK_GENERIC_CPU_AND_GPU_H
#define POINTWALK_GENERIC_CPU_AND_GPU_H

// HD: compile a function for both CPU and GPU (__host__ __device__ under nvcc,
// nothing under g++).
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#endif
