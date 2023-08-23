#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, cublasDiagType_t unit_nonunit, int n, int k,
          const cuDoubleComplex *a, int lda, cuDoubleComplex *x, int incx) {
  // Start
  cublasZtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
              n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
              x /*cuDoubleComplex **/, incx /*int*/);
  // End
}
