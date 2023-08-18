#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, cublasDiagType_t unit_nonunit, int n, int k,
          const cuComplex *a, int lda, cuComplex *x, int incx) {
  // Start
  cublasCtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
              n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
              x /*cuComplex **/, incx /*int*/);
  // End
}
