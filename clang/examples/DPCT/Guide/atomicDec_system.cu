__global__ void test(unsigned *pu, unsigned u) {
  // Start
  atomicDec_system(pu /*unsigned **/, u /*unsigned*/);
  // End
}
