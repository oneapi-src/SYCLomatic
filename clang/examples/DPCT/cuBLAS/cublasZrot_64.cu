#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx,
          cuDoubleComplex *y, int64_t incy, const double *c,
          const cuDoubleComplex *s) {
  // Start
  cublasZrot_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                x /*cuDoubleComplex **/, incx /*int64_t*/,
                y /*cuDoubleComplex **/, incy /*int64_t*/, c /*const double **/,
                s /*const cuDoubleComplex **/);
  // End
}
