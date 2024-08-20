#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, const double *a, int lda, const double *tau) {
  // Start
  int buffer_size;
  cusolverDnDorgbr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
      tau /*const double **/, &buffer_size /*int **/);
  // End
}
