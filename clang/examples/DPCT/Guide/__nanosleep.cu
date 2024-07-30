__global__ void test(unsigned u) {
  // Start
  __nanosleep(u /*unsigned*/);
  // End
}
