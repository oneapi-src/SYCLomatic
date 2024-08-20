#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const float *a, int lda, const float *d, const float *e,
          const float *tau) {
  // Start
  int buffer_size;
  cusolverDnSsytrd_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const float **/, lda /*int*/, d /*const float **/,
      e /*const float **/, tau /*const float **/, &buffer_size /*int **/);
  // End
}
