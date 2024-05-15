#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a,
          int lda, float *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnSpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*float **/, lda /*int*/, buffer /*float **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
