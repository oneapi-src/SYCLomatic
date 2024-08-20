#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *a,
          int lda) {
  // Start
  int buffer_size;
  cusolverDnZgetrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
