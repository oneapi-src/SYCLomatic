#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
          const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
  // Start
  cublasZaxpy(handle /*cublasHandle_t*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
              incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/);
  // End
}
