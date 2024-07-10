__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     long long *pll, long long ll) {
  // Start
  atomicMin_system(pi /*int **/, i /*int*/);
  atomicMin_system(pu /*unsigned **/, u /*unsigned*/);
  atomicMin_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicMin_system(pll /*long long **/, ll /*long long*/);
  // End
}
