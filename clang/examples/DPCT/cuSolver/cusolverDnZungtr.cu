#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          cuDoubleComplex *a, int lda, const cuDoubleComplex *tau,
          cuDoubleComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnZungtr(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*cuDoubleComplex **/, lda /*int*/, tau /*const cuDoubleComplex **/,
      buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
