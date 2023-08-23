#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx,
          const cuDoubleComplex *y, int incy, cuDoubleComplex *a, int lda) {
  // Start
  cublasZsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const cuDoubleComplex **/,
              x /*const cuDoubleComplex **/, incx /*int*/,
              y /*const cuDoubleComplex **/, incy /*int*/,
              a /*cuDoubleComplex **/, lda /*int*/);
  // End
}
