#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
          const cuDoubleComplex *a, int lda, cuDoubleComplex *b, int ldb,
          int *info) {
  // Start
  cusolverDnZpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, nrhs /*int*/, a /*const cuDoubleComplex **/,
                   lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/,
                   info /*int **/);
  // End
}
