#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, const double *a,
          int lda, const double *tau) {
  // Start
  int buffer_size;
  cusolverDnDorgqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, k /*int*/, a /*const double **/,
                              lda /*int*/, tau /*const double **/,
                              &buffer_size /*int **/);
  // End
}
