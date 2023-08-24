#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const float *const *a, int lda, const int *ipiv, float *const *b,
          int ldb, int *info, int group_count) {
  // Start
  cublasSgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                      n /*int*/, nrhs /*int*/, a /*const float *const **/,
                      lda /*int*/, ipiv /*const int **/, b /*float *const **/,
                      ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
