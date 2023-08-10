#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const double *alpha, const cuDoubleComplex *x, int incx,
          cuDoubleComplex *a, int lda) {
  // Start
  cublasZher(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
             n /*int*/, alpha /*const double **/, x /*const cuDoubleComplex **/,
             incx /*int*/, a /*cuDoubleComplex **/, lda /*int*/);
  // End
}
