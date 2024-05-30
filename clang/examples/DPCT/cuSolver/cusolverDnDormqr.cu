#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const double *a,
          int lda, const double *tau, double *c, int ldc, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDormqr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const double **/, lda /*int*/, tau /*const double **/, c /*double **/,
      ldc /*int*/, buffer /*double **/, buffer_size /*int*/, info /*int **/);
  // End
}
