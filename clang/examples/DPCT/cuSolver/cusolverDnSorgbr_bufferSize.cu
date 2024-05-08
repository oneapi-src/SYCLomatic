#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, const float *a, int lda, const float *tau) {
  // Start
  int buffer_size;
  cusolverDnSorgbr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
      tau /*const float **/, &buffer_size /*int **/);
  // End
}
