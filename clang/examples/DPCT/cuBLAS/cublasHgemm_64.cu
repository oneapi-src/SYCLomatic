#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
          const __half *alpha, const __half *a, int64_t lda, const __half *b,
          int64_t ldb, const __half *beta, __half *c, int64_t ldc) {
  // Start
  cublasHgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 k /*int64_t*/, alpha /*const __half **/, a /*const __half **/,
                 lda /*int64_t*/, b /*const __half **/, ldb /*int64_t*/,
                 beta /*const __half **/, c /*__half **/, ldc /*int64_t*/);
  // End
}
