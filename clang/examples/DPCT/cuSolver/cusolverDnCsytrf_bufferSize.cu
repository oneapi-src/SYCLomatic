#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int n, cuComplex *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnCsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
                              a /*cuComplex **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
