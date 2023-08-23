#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
          const cuComplex *alpha, const cuComplex *a, int lda,
          const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y,
          int incy) {
  // Start
  cublasCgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
              n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
              lda /*int*/, x /*const cuComplex **/, incx /*int*/,
              beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
  // End
}
