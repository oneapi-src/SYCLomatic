__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicOr_system(pi /*int **/, i /*int*/);
  atomicOr_system(pu /*unsigned **/, u /*unsigned*/);
  atomicOr_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
