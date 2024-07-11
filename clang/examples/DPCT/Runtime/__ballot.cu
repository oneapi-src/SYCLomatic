__device__ void test(unsigned int r, int pred) {
  // Start
  r = __ballot(pred /*int*/);
  // End
}