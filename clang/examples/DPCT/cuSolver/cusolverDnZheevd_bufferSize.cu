#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, const cuDoubleComplex *a, int lda,
          const double *w) {
  // Start
  int buffer_size;
  cusolverDnZheevd_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
      uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuDoubleComplex **/,
      lda /*int*/, w /*const double **/, &buffer_size /*int **/);
  // End
}
