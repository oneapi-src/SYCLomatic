#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a,
          int lda, float *d, float *e, float *tau, float *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnSsytrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*float **/, lda /*int*/, d /*float **/,
                   e /*float **/, tau /*float **/, buffer /*float **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
