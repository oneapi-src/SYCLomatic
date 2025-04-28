#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t m, int64_t n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx,
          const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *a,
          int64_t lda) {
  // Start
  cublasZgeru_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuDoubleComplex **/,
                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
                 y /*const cuDoubleComplex **/, incy /*int64_t*/,
                 a /*cuDoubleComplex **/, lda /*int64_t*/);
  // End
}
