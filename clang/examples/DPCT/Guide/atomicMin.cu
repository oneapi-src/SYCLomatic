__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     long long *pll, long long ll) {
  // Start
  atomicMin(pi /*int **/, i /*int*/);
  atomicMin(pu /*unsigned **/, u /*unsigned*/);
  atomicMin(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicMin(pll /*long long **/, ll /*long long*/);
  // End
}
