__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     long long *pll, long long ll) {
  // Start
  atomicMax(pi /*int **/, i /*int*/);
  atomicMax(pu /*unsigned **/, u /*unsigned*/);
  atomicMax(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicMax(pll /*long long **/, ll /*long long*/);
  // End
}
