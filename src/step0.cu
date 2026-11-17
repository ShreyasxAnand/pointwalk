__device__ void atomicMinFloat(float*address, float val) {
    atomicMin((int*)address, __float_as_int(val));
}

__device__ void atomicMinFloatGeneric(float*address, float val) {
    int *a = (int*)address;
    int old = *a;
    while (val < __int_as_float(old)) {
        int prev = atomicCAS(a, old, __float_as_int(val));
        if (prev == old) break;
        old = prev;
    }
}
