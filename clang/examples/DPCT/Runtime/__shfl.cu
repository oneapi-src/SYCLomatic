__device__ void test(unsigned int r, unsigned int var, int src_lane,
                     int width) {
  // Start
  r = __shfl(var /*unsigned int*/, src_lane /*int*/, width /*int*/);
  // End
}