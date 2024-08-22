__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     unsigned short *pus, float *pf, float f, double *pd,
                     double d) {
  // Start
  atomicAdd_system(pi /*int **/, i /*int*/);
  atomicAdd_system(pu /*unsigned **/, u /*unsigned*/);
  atomicAdd_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicAdd_system(pf /*float **/, f /*float*/);
  atomicAdd_system(pd /*double **/, d /*double*/);
  // End
}
