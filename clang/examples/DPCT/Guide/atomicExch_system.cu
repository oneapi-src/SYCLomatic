__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     float *pf, float f) {
  // Start
  atomicExch_system(pi /*int **/, i /*int*/);
  atomicExch_system(pu /*unsigned **/, u /*unsigned*/);
  atomicExch_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicExch_system(pf /*float **/, f /*float*/);
  // End
}
