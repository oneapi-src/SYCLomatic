__global__ void test(unsigned *pu, unsigned u) {
  // Start
  atomicInc(pu /*unsigned **/, u /*unsigned*/);
  // End
}
