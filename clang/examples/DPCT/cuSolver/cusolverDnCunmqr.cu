#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const cuComplex *a,
          int lda, const cuComplex *tau, cuComplex *c, int ldc,
          cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCunmqr(handle /*cusolverDnHandle_t*/,
                   left_right /*cublasSideMode_t*/, trans /*cublasOperation_t*/,
                   m /*int*/, n /*int*/, k /*int*/, a /*const cuComplex **/,
                   lda /*int*/, tau /*const cuComplex **/, c /*cuComplex **/,
                   ldc /*int*/, buffer /*cuComplex **/, buffer_size /*int*/,
                   info /*int **/);
  // End
}
