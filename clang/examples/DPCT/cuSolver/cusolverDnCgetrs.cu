#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const cuComplex *a, int lda, const int *ipiv, cuComplex *b, int ldb,
          int *info) {
  // Start
  cusolverDnCgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
                   n /*int*/, nrhs /*int*/, a /*const cuComplex **/,
                   lda /*int*/, ipiv /*const int **/, b /*cuComplex **/,
                   ldb /*int*/, info /*int **/);
  // End
}
