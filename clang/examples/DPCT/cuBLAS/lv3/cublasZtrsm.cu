#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int lda,
          cuDoubleComplex *b, int ldb) {
  // Start
  cublasZtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
              unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
              lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/);
  // End
}
