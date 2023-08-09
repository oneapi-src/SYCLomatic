#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const float *alpha,
          const float *const *a, int lda, const float *const *b, int ldb,
          const float *beta, float *const *c, int ldc, int group_count) {
  // Start
  cublasSgemmBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const float **/, a /*const float *const **/, lda /*int*/,
      b /*const float *const **/, ldb /*int*/, beta /*const float **/,
      c /*float *const **/, ldc /*int*/, group_count /*int*/);
  // End
}
