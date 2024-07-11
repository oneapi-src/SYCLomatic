#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int n, double *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnDsytrf_bufferSize(handle /*cusolverDnHandle_t*/, n /*int*/,
                              a /*double **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
