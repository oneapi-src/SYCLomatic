#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasOperation_t trans, int m, int n, int k, const double *a,
          int lda, const double *tau, const double *c, int ldc) {
  // Start
  int buffer_size;
  cusolverDnDormqr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/,
      trans /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const double **/, lda /*int*/, tau /*const double **/,
      c /*const double **/, ldc /*int*/, &buffer_size /*int **/);
  // End
}
