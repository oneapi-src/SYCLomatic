#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *tau,
          cuDoubleComplex *c, int ldc, cuDoubleComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnZunmqr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/,
      tau /*const cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/,
      buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
