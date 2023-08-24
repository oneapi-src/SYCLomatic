#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl,
          int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *a,
          int lda, const cuDoubleComplex *x, int incx,
          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
  // Start
  cublasZgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
              n /*int*/, kl /*int*/, ku /*int*/,
              alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
              lda /*int*/, x /*const cuDoubleComplex **/, incx /*int*/,
              beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
              incy /*int*/);
  // End
}
