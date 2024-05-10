#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const cuDoubleComplex *a, int lda, const double *d, const double *e,
          const cuDoubleComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnZhetrd_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/, d /*const double **/,
      e /*const double **/, tau /*const cuDoubleComplex **/,
      &buffer_size /*int **/);
  // End
}
