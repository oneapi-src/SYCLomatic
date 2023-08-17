#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *a,
          int lda) {
  // Start
  cublasCsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
             n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
             incx /*int*/, a /*cuComplex **/, lda /*int*/);
  // End
}
