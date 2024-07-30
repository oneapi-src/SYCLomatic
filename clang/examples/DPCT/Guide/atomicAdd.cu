#include "cuda_fp16.h"

__global__ void test(int *pi, int i, unsigned *pu, unsigned u,
                     unsigned long long *pull, unsigned long long ull,
                     unsigned short *pus, float *pf, float f, double *pd,
                     double d, __half2 *ph2, __half2 h2) {
  // Start
  atomicAdd(pi /*int **/, i /*int*/);
  atomicAdd(pu /*unsigned **/, u /*unsigned*/);
  atomicAdd(pull /*unsigned long long **/, ull /*unsigned long long*/);
  atomicAdd(pf /*float **/, f /*float*/);
  atomicAdd(pd /*double **/, d /*double*/);
  atomicAdd(ph2 /*__half2 **/, h2 /*__half2*/);
  // End
}
