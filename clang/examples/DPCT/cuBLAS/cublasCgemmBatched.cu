#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
          const cuComplex *const *a, int lda, const cuComplex *const *b,
          int ldb, const cuComplex *beta, cuComplex *const *c, int ldc,
          int group_count) {
  // Start
  cublasCgemmBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuComplex **/, a /*const cuComplex *const **/, lda /*int*/,
      b /*const cuComplex *const **/, ldb /*int*/, beta /*const cuComplex **/,
      c /*cuComplex *const **/, ldc /*int*/, group_count /*int*/);
  // End
}
