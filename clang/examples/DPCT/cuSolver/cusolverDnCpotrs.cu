#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
          const cuComplex *a, int lda, cuComplex *b, int ldb, int *info) {
  // Start
  cusolverDnCpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, nrhs /*int*/, a /*const cuComplex **/,
                   lda /*int*/, b /*cuComplex **/, ldb /*int*/, info /*int **/);
  // End
}
