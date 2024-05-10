#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, double *a, int lda) {
  // Start
  int buffer_size;
  cusolverDnDgeqrf_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, a /*double **/, lda /*int*/,
                              &buffer_size /*int **/);
  // End
}
