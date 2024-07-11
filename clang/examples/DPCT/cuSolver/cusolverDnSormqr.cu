#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const float *a, int lda,
          const float *tau, float *c, int ldc, float *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnSormqr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const float **/, lda /*int*/, tau /*const float **/, c /*float **/,
      ldc /*int*/, buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
