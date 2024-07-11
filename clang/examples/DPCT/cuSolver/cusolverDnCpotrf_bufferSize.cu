#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *a,
          int lda) {
  // Start
  int buffer_size;
  cusolverDnCpotrf_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*cuComplex **/, lda /*int*/, &buffer_size /*int **/);
  // End
}
