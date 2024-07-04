__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     float *pf, float f) {
  // Start
  atomicExch(pi /*int **/, i /*int*/);
  atomicExch(pu /*unsigned **/, u /*unsigned*/);
  atomicExch(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicExch(pf /*float **/, f /*float*/);
  // End
}
