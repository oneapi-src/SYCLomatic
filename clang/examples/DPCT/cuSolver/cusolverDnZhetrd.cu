#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          cuDoubleComplex *a, int lda, double *d, double *e,
          cuDoubleComplex *tau, cuDoubleComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnZhetrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
                   d /*double **/, e /*double **/, tau /*cuDoubleComplex **/,
                   buffer /*cuDoubleComplex **/, buffer_size /*int*/,
                   info /*int **/);
  // End
}
