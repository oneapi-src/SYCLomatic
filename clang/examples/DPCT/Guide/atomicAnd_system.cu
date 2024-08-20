__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull) {
  // Start
  atomicAnd_system(pi /*int **/, i /*int*/);
  atomicAnd_system(pu /*unsigned **/, u /*unsigned*/);
  atomicAnd_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  // End
}
