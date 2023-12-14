__device__ void test(unsigned int r, unsigned int mask, int value) {
  // Start
  r = __match_any_sync(mask /*unsigned int*/, value /*int*/);
  // End
}