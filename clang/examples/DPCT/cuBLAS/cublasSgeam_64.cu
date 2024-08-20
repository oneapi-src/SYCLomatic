#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, const float *alpha,
          const float *a, int64_t lda, const float *beta, const float *b,
          int64_t ldb, float *c, int64_t ldc) {
  // Start
  cublasSgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
                 beta /*const float **/, b /*const float **/, ldb /*int64_t*/,
                 c /*float **/, ldc /*int64_t*/);
  // End
}
