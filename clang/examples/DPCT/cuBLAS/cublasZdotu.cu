#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
          const cuDoubleComplex *y, int incy, cuDoubleComplex *res) {
  // Start
  cublasZdotu(handle /*cublasHandle_t*/, n /*int*/,
              x /*const cuDoubleComplex **/, incx /*int*/,
              y /*const cuDoubleComplex **/, incy /*int*/,
              res /*cuDoubleComplex **/);
  // End
}
