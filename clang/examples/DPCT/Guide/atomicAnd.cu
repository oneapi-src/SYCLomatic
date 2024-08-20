__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicAnd(pi /*int **/, i /*int*/);
  atomicAnd(pu /*unsigned **/, u /*unsigned*/);
  atomicAnd(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
