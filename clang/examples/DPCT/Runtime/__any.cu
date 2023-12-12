__device__ void test(unsigned int r, int pred) {
  // Start
  r = __any(pred /*int*/);
  // End
}