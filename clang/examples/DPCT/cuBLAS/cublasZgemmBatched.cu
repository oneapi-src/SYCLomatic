#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k,
          const cuDoubleComplex *alpha, const cuDoubleComplex *const *a,
          int lda, const cuDoubleComplex *const *b, int ldb,
          const cuDoubleComplex *beta, cuDoubleComplex *const *c, int ldc,
          int group_count) {
  // Start
  cublasZgemmBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex *const **/,
      lda /*int*/, b /*const cuDoubleComplex *const **/, ldb /*int*/,
      beta /*const cuDoubleComplex **/, c /*cuDoubleComplex *const **/,
      ldc /*int*/, group_count /*int*/);
  // End
}
