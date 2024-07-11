#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
          cuDoubleComplex *a, int lda, cuDoubleComplex *tau, cuDoubleComplex *c,
          int ldc, cuDoubleComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnZunmtr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
      n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
      tau /*cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/,
      buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
