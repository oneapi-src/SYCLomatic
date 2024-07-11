__global__ void test(unsigned *pu, unsigned u) {
  // Start
  atomicDec(pu /*unsigned **/, u /*unsigned*/);
  // End
}
