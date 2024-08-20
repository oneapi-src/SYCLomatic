__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicXor_system(pi /*int **/, i /*int*/);
  atomicXor_system(pu /*unsigned **/, u /*unsigned*/);
  atomicXor_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
