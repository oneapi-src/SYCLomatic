#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const double *a, int lda, const int *ipiv, double *b, int ldb,
          int *info) {
  // Start
  cusolverDnDgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
                   n /*int*/, nrhs /*int*/, a /*const double **/, lda /*int*/,
                   ipiv /*const int **/, b /*double **/, ldb /*int*/,
                   info /*int **/);
  // End
}
