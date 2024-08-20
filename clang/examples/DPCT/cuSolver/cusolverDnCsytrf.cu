#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *a,
          int lda, int *ipiv, cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCsytrf(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuComplex **/, lda /*int*/, ipiv /*int **/,
                   buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
