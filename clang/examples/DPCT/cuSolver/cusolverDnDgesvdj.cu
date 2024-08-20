#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, double *a, int lda, double *s, double *u, int ldu, double *v,
          int ldv, double *buffer, int buffer_size, int *info,
          gesvdjInfo_t params) {
  // Start
  cusolverDnDgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    econ /*int*/, m /*int*/, n /*int*/, a /*double **/,
                    lda /*int*/, s /*double **/, u /*double **/, ldu /*int*/,
                    v /*double **/, ldv /*int*/, buffer /*double **/,
                    buffer_size /*int*/, info /*int **/,
                    params /*gesvdjInfo_t*/);
  // End
}