#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const float *alpha, const cuComplex *x, int64_t incx, cuComplex *a) {
  // Start
  cublasChpr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                n /*int64_t*/, alpha /*const float **/, x /*const cuComplex **/,
                incx /*int64_t*/, a /*cuComplex **/);
  // End
}
