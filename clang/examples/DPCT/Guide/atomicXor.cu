__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicXor(pi /*int **/, i /*int*/);
  atomicXor(pu /*unsigned **/, u /*unsigned*/);
  atomicXor(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
