#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int m, int n, const cuComplex *alpha,
          const cuComplex *a, int lda, const cuComplex *b, int ldb,
          cuComplex *c, int ldc) {
  // Start
  cublasCtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
              unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
              alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
              b /*const cuComplex **/, ldb /*int*/, c /*cuComplex **/,
              ldc /*int*/);
  // End
}
