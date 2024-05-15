#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const cuDoubleComplex *a, int lda, const int *ipiv,
          cuDoubleComplex *b, int ldb, int *info) {
  // Start
  cusolverDnZgetrs(handle /*cusolverDnHandle_t*/, trans /*cublasOperation_t*/,
                   n /*int*/, nrhs /*int*/, a /*const cuDoubleComplex **/,
                   lda /*int*/, ipiv /*const int **/, b /*cuDoubleComplex **/,
                   ldb /*int*/, info /*int **/);
  // End
}
