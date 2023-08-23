#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
          cuDoubleComplex *x, int incx) {
  // Start
  cublasZscal(handle /*cublasHandle_t*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
              incx /*int*/);
  // End
}
