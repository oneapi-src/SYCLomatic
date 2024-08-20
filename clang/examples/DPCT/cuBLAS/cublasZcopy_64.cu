#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x,
          int64_t incx, cuDoubleComplex *y, int64_t incy) {
  // Start
  cublasZcopy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 x /*const cuDoubleComplex **/, incx /*int64_t*/,
                 y /*cuDoubleComplex **/, incy /*int64_t*/);
  // End
}
