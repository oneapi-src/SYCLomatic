__global__ void test(int *pi, int i, unsigned *pu, unsigned u) {
  // Start
  atomicSub_system(pi /*int **/, i /*int*/);
  atomicSub_system(pu /*unsigned **/, u /*unsigned*/);
  // End
}
