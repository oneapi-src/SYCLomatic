#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, double *a, int lda, const double *tau, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDorgbr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*double **/, lda /*int*/, tau /*const double **/,
      buffer /*double **/, buffer_size /*int*/, info /*int **/);
  // End
}
