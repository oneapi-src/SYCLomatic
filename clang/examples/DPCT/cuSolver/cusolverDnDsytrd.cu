#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a,
          int lda, double *d, double *e, double *tau, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDsytrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*double **/, lda /*int*/, d /*double **/,
                   e /*double **/, tau /*double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
