#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, float *a, int lda,
          float *tau, float *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnSgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*float **/, lda /*int*/, tau /*float **/,
                   buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
