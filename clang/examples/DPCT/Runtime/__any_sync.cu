__device__ void test(unsigned int r, unsigned int mask, int pred) {
  // Start
  r = __any_sync(mask /*unsigned int*/, pred /*int*/);
  // End
}