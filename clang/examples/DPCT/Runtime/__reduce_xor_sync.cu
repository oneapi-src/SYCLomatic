__device__ void test(unsigned int r, unsigned int mask, unsigned int value) {
  // Start
  r = __reduce_xor_sync(mask /*unsigned int*/, value /*unsigned int*/);
  // End
}