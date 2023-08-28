#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const float *alpha, const cuComplex *x, int incx, cuComplex *a) {
  // Start
  cublasChpr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
             n /*int*/, alpha /*const float **/, x /*const cuComplex **/,
             incx /*int*/, a /*cuComplex **/);
  // End
}
