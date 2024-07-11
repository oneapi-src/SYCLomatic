#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const double *a, int lda, const double *d, const double *e,
          const double *tau) {
  // Start
  int buffer_size;
  cusolverDnDsytrd_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const double **/, lda /*int*/, d /*const double **/,
      e /*const double **/, tau /*const double **/, &buffer_size /*int **/);
  // End
}
