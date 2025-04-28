#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const double *alpha, const cuDoubleComplex *x, int64_t incx,
          cuDoubleComplex *a) {
  // Start
  cublasZhpr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                n /*int64_t*/, alpha /*const double **/,
                x /*const cuDoubleComplex **/, incx /*int64_t*/,
                a /*cuDoubleComplex **/);
  // End
}
