#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          cuDoubleComplex *a, int lda, cuDoubleComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnZpotrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
                   buffer /*cuDoubleComplex **/, buffer_size /*int*/,
                   info /*int **/);
  // End
}
