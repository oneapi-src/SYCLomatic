#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, const float *a, int lda,
          const float *w) {
  // Start
  int buffer_size;
  cusolverDnSsyevd_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
      uplo /*cublasFillMode_t*/, n /*int*/, a /*const float **/, lda /*int*/,
      w /*const float **/, &buffer_size /*int **/);
  // End
}
