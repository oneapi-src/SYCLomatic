#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int64_t m, int64_t n, const float *alpha,
          const float *a, int64_t lda, float *b, int64_t ldb) {
  // Start
  cublasStrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
                 unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
                 b /*float **/, ldb /*int64_t*/);
  // End
}
