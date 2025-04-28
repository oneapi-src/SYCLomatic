#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx,
          cuDoubleComplex *a, int64_t lda) {
  // Start
  cublasZsyr_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                n /*int64_t*/, alpha /*const cuDoubleComplex **/,
                x /*const cuDoubleComplex **/, incx /*int64_t*/,
                a /*cuDoubleComplex **/, lda /*int64_t*/);
  // End
}
