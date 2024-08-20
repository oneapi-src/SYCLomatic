#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, float *a, int lda, const float *tau, float *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnSorgbr(handle /*cusolverDnHandle_t*/,
                   left_right /*cublasSideMode_t*/, m /*int*/, n /*int*/,
                   k /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
                   buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
