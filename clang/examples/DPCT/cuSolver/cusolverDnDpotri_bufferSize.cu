#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a,
          int lda) {
  // Start
  int buffer_size;
  cusolverDnDpotri_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*double **/, lda /*int*/, &buffer_size /*int **/);
  // End
}
