#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const float *a, int lda, const int *ipiv, float *b, int ldb,
          int *info) {
  // Start
  cusolverDnSgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
                   n /*int*/, nrhs /*int*/, a /*const float **/, lda /*int*/,
                   ipiv /*const int **/, b /*float **/, ldb /*int*/,
                   info /*int **/);
  // End
}
