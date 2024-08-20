#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *a,
          int lda, const cuComplex *tau, cuComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnCungtr(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuComplex **/, lda /*int*/,
                   tau /*const cuComplex **/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
