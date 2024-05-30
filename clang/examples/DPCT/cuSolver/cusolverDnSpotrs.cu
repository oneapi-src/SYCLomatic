#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
          const float *a, int lda, float *b, int ldb, int *info) {
  // Start
  cusolverDnSpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, nrhs /*int*/, a /*const float **/, lda /*int*/,
                   b /*float **/, ldb /*int*/, info /*int **/);
  // End
}
