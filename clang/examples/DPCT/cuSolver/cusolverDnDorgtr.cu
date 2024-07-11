#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a,
          int lda, const double *tau, double *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnDorgtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*double **/, lda /*int*/,
                   tau /*const double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
