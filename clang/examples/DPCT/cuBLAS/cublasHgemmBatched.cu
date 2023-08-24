#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const __half *alpha,
          const __half *const *a, int lda, const __half *const *b, int ldb,
          const __half *beta, __half *const *c, int ldc, int group_count) {
  // Start
  cublasHgemmBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const __half **/, a /*const __half *const **/, lda /*int*/,
      b /*const __half *const **/, ldb /*int*/, beta /*const __half **/,
      c /*__half *const **/, ldc /*int*/, group_count /*int*/);
  // End
}
