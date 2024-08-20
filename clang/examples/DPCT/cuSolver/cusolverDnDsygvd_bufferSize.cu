#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *a,
          int lda, const double *b, int ldb, const double *w) {
  // Start
  int buffer_size;
  cusolverDnDsygvd_bufferSize(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const double **/, lda /*int*/, b /*const double **/, ldb /*int*/,
      w /*const double **/, &buffer_size /*int **/);
  // End
}
