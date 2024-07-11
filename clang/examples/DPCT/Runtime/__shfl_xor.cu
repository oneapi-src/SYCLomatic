__device__ void test(unsigned int r, unsigned int var, int lane, int width) {
  // Start
  r = __shfl_xor(var /*unsigned int*/, lane /*int*/, width /*int*/);
  // End
}