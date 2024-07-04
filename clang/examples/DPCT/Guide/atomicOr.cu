__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicOr(pi /*int **/, i /*int*/);
  atomicOr(pu /*unsigned **/, u /*unsigned*/);
  atomicOr(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
