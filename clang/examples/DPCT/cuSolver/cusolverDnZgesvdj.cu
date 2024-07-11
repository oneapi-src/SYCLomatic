#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, cuDoubleComplex *a, int lda, double *s, cuDoubleComplex *u,
          int ldu, cuDoubleComplex *v, int ldv, cuDoubleComplex *buffer,
          int buffer_size, int *info, gesvdjInfo_t params) {
  // Start
  cusolverDnZgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    econ /*int*/, m /*int*/, n /*int*/, a /*cuDoubleComplex **/,
                    lda /*int*/, s /*double **/, u /*cuDoubleComplex **/,
                    ldu /*int*/, v /*cuDoubleComplex **/, ldv /*int*/,
                    buffer /*cuDoubleComplex **/, buffer_size /*int*/,
                    info /*int **/, params /*gesvdjInfo_t*/);
  // End
}
