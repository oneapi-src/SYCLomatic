#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, cublasOperation_t transa,
          cublasDiagType_t unit_diag, int64_t m, int64_t n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int64_t lda,
          cuDoubleComplex *b, int64_t ldb) {
  // Start
  cublasZtrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
                 unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuDoubleComplex **/,
                 a /*const cuDoubleComplex **/, lda /*int64_t*/,
                 b /*cuDoubleComplex **/, ldb /*int64_t*/);
  // End
}
