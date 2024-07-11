#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
          const double *a, int lda, double *b, int ldb, int *info) {
  // Start
  cusolverDnDpotrs(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, nrhs /*int*/, a /*const double **/, lda /*int*/,
                   b /*double **/, ldb /*int*/, info /*int **/);
  // End
}
