#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, cuDoubleComplex *a, int lda, double *w,
          cuDoubleComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnZheevd(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                   uplo /*cublasFillMode_t*/, n /*int*/,
                   a /*cuDoubleComplex **/, lda /*int*/, w /*double **/,
                   buffer /*cuDoubleComplex **/, buffer_size /*int*/,
                   info /*int **/);
  // End
}
