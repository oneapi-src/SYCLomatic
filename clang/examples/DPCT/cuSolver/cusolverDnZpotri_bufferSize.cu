#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          cuDoubleComplex *a, int lda, int *Lwork) {
  // Start
  int buffer_size;
  cusolverDnZpotri_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*cuDoubleComplex **/, lda /*int*/, &buffer_size /*int **/);
  // End
}
