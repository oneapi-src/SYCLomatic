#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
          int64_t kl, int64_t ku, const cuDoubleComplex *alpha,
          const cuDoubleComplex *a, int64_t lda, const cuDoubleComplex *x,
          int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y,
          int64_t incy) {
  // Start
  cublasZgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                 m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
                 alpha /*const cuDoubleComplex **/,
                 a /*const cuDoubleComplex **/, lda /*int64_t*/,
                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
                 beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
                 incy /*int64_t*/);
  // End
}
