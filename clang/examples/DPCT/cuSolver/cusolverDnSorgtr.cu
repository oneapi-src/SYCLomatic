#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a,
          int lda, const float *tau, float *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnSorgtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*float **/, lda /*int*/, tau /*const float **/,
                   buffer /*float **/, buffer_size /*int*/, info /*int **/);
  // End
}
