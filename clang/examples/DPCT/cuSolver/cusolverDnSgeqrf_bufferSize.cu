#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, float *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnSgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, a /*float **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
