#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *a,
          int lda, double *b, int ldb, double *w, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDsygvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
                   jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*double **/, lda /*int*/, b /*double **/,
                   ldb /*int*/, w /*double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
