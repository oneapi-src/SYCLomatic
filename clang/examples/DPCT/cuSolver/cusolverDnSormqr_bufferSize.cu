#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const float *a, int lda,
          const float *tau, const float *c, int ldc) {
  // Start
  int buffer_size;
  cusolverDnSormqr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const float **/, lda /*int*/, tau /*const float **/,
      c /*const float **/, ldc /*int*/, &buffer_size /*int **/);
  // End
}
