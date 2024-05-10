#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuComplex *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnCgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, a /*cuComplex **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
