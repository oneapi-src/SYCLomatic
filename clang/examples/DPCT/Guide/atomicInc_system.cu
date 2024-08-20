__global__ void test(unsigned *pu, unsigned u) {
  // Start
  atomicInc_system(pu /*unsigned **/, u /*unsigned*/);
  // End
}
