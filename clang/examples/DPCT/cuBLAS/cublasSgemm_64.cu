#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
          const float *alpha, const float *a, int64_t lda, const float *b,
          int64_t ldb, const float *beta, float *c, int64_t ldc) {
  // Start
  cublasSgemm_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 k /*int64_t*/, alpha /*const float **/, a /*const float **/,
                 lda /*int64_t*/, b /*const float **/, ldb /*int64_t*/,
                 beta /*const float **/, c /*float **/, ldc /*int64_t*/);
  // End
}
