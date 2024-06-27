#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const cuComplex *a,
          int lda, const cuComplex *tau, const cuComplex *c, int ldc) {
  // Start
  int buffer_size;
  cusolverDnCunmqr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const cuComplex **/, lda /*int*/, tau /*const cuComplex **/,
      c /*const cuComplex **/, ldc /*int*/, &buffer_size /*int **/);
  // End
}
