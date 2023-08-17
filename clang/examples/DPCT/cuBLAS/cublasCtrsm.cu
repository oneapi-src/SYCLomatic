#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n, const cuComplex *alpha,
          const cuComplex *a, int lda, cuComplex *b, int ldb) {
  // Start
  cublasCtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
              unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
              alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
              b /*cuComplex **/, ldb /*int*/);
  // End
}
