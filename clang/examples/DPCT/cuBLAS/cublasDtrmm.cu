#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n, const double *alpha,
          const double *a, int lda, const double *b, int ldb, double *c,
          int ldc) {
  // Start
  cublasDtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
              unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
              alpha /*const double **/, a /*const double **/, lda /*int*/,
              b /*const double **/, ldb /*int*/, c /*double **/, ldc /*int*/);
  // End
}
