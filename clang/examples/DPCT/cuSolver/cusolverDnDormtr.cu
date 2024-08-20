#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
          double *a, int lda, double *tau, double *c, int ldc, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDormtr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
      n /*int*/, a /*double **/, lda /*int*/, tau /*double **/, c /*double **/,
      ldc /*int*/, buffer /*double **/, buffer_size /*int*/, info /*int **/);
  // End
}
