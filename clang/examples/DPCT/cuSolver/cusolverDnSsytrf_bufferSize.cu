#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int n, float *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnSsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
                              a /*float **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
