#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
          const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
          int incy, cuDoubleComplex *a, int lda) {
  // Start
  cublasZgerc(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
              incx /*int*/, y /*const cuDoubleComplex **/, incy /*int*/,
              a /*cuDoubleComplex **/, lda /*int*/);
  // End
}
