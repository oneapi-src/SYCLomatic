__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     long long *pll, long long ll) {
  // Start
  atomicMax_system(pi /*int **/, i /*int*/);
  atomicMax_system(pu /*unsigned **/, u /*unsigned*/);
  atomicMax_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicMax_system(pll /*long long **/, ll /*long long*/);
  // End
}
