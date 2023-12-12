__device__ void test(unsigned int r, unsigned int var, unsigned int delta,
                     int width) {
  // Start
  r = __shfl_down(var /*unsigned int*/, delta /*unsigned int*/, width /*int*/);
  // End
}