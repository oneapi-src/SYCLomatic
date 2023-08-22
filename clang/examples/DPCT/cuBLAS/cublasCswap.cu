#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y,
          int incy) {
  // Start
  cublasCswap(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
              incx /*int*/, y /*cuComplex **/, incy /*int*/);
  // End
}
