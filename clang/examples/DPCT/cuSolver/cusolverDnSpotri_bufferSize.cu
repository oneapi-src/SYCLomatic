#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a,
          int lda) {
  // Start
  int buffer_size;
  cusolverDnSpotri_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*float **/, lda /*int*/, &buffer_size /*int **/);
  // End
}
