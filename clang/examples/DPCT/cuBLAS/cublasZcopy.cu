#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
          cuDoubleComplex *y, int incy) {
  // Start
  cublasZcopy(handle /*cublasHandle_t*/, n /*int*/,
              x /*const cuDoubleComplex **/, incx /*int*/,
              y /*cuDoubleComplex **/, incy /*int*/);
  // End
}
