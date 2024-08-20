#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
          float *a, int lda, float *tau, float *c, int ldc, float *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnSormtr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
      n /*int*/, a /*float **/, lda /*int*/, tau /*float **/, c /*float **/,
      ldc /*int*/, buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
