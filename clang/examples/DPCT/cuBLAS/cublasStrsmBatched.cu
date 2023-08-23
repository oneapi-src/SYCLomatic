#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n, const float *alpha,
          const float *const *a, int lda, float *const *b, int ldb,
          int group_count) {
  // Start
  cublasStrsmBatched(
      handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
      upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
      unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
      alpha /*const float **/, a /*const float *const **/, lda /*int*/,
      b /*float *const **/, ldb /*int*/, group_count /*int*/);
  // End
}
