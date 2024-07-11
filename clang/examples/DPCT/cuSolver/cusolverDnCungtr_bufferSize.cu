#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const cuComplex *a, int lda, const cuComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnCungtr_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuComplex **/, lda /*int*/, tau /*const cuComplex **/,
      &buffer_size /*int **/);
  // End
}
