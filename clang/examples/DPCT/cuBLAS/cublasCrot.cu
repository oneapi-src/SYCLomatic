#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y,
          int incy, const float *c, const cuComplex *s) {
  // Start
  cublasCrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
             incx /*int*/, y /*cuComplex **/, incy /*int*/, c /*const float **/,
             s /*const cuComplex **/);
  // End
}
