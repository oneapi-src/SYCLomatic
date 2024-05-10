#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, const float *a,
          int lda, const float *tau) {
  // Start
  int buffer_size;
  cusolverDnSorgqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, k /*int*/, a /*const float **/,
                              lda /*int*/, tau /*const float **/,
                              &buffer_size /*int **/);
  // End
}
