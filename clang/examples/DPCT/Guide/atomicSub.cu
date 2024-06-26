__global__ void test(int *pi, int i, unsigned *pu, unsigned u) {
  // Start
  atomicSub(pi /*int **/, i /*int*/);
  atomicSub(pu /*unsigned **/, u /*unsigned*/);
  // End
}
