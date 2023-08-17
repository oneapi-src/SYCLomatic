#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int lda,
          const cuDoubleComplex *b, int ldb, const cuDoubleComplex *beta,
          cuDoubleComplex *c, int ldc) {
  // Start
  cublasZgemm3m(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
      lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
      beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/);
  // End
}
