#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int n, cuDoubleComplex *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnZsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
                              a /*cuDoubleComplex **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
