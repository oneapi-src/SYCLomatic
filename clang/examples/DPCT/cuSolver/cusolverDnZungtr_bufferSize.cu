#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnZungtr_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/,
      tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
  // End
}
