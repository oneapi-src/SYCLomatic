#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
          const float *a, int lda, const float *tau, const float *c, int ldc) {
  // Start
  int buffer_size;
  cusolverDnSormtr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      uplo /*cublasFillMode_t*/, trans /*cublasOperation_t*/, m /*int*/,
      n /*int*/, a /*const float **/, lda /*int*/, tau /*const float **/,
      c /*const float **/, ldc /*int*/, &buffer_size /*int **/);
  // End
}
