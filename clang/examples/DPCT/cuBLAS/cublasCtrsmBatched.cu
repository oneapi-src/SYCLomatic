#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n, const cuComplex *alpha,
          const cuComplex *const *a, int lda, cuComplex *const *b, int ldb,
          int group_count) {
  // Start
  cublasCtrsmBatched(
      handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
      upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
      unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
      alpha /*const cuComplex **/, a /*const cuComplex *const **/, lda /*int*/,
      b /*cuComplex *const **/, ldb /*int*/, group_count /*int*/);
  // End
}
