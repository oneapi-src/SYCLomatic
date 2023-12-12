__device__ void test(unsigned int r, unsigned int mask, int value, int *pred) {
  // Start
  r = __match_all_sync(mask /*unsigned int*/, value /*int*/, pred /*int*/);
  // End
}