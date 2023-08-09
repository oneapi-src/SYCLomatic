#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const cuComplex *alpha, const cuComplex *a, int lda,
          const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y,
          int incy) {
  // Start
  cublasSsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
              lda /*int*/, x /*const cuComplex **/, incx /*int*/,
              beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
  // End
}
