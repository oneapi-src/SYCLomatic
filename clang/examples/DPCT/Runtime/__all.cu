__device__ void test(unsigned int r, int pred) {
  // Start
  r = __all(pred /*int*/);
  // End
}