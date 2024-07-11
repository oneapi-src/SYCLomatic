#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
          const cuComplex *alpha, const cuComplex *a, int64_t lda,
          const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y,
          int64_t incy) {
  // Start
  cublasCgemv_64(
      handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int64_t*/,
      n /*int64_t*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
      lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
      beta /*const cuComplex **/, y /*cuComplex **/, incy /*int64_t*/);
  // End
}
