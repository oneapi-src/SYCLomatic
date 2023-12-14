__device__ void test(unsigned int r, unsigned int mask, unsigned int var,
                     int lane, int width) {
  // Start
  r = __shfl_xor_sync(mask /*unsigned int*/, var /*unsigned int*/, lane /*int*/,
                      width /*int*/);
  // End
}