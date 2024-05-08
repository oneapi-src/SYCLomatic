#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, float *a, int lda,
          const float *tau, float *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnSorgqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   k /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
                   buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
