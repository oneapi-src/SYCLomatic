#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, float *a, int lda, float *s, float *u, int ldu, float *v,
          int ldv, float *buffer, int buffer_size, int *info,
          gesvdjInfo_t params) {
  // Start
  cusolverDnSgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    econ /*int*/, m /*int*/, n /*int*/, a /*float **/,
                    lda /*int*/, s /*float **/, u /*float **/, ldu /*int*/,
                    v /*float **/, ldv /*int*/, buffer /*float **/,
                    buffer_size /*int*/, info /*int **/,
                    params /*gesvdjInfo_t*/);
  // End
}
