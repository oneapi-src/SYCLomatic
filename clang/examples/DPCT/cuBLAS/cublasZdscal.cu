#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x,
          int incx) {
  // Start
  cublasZdscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
               x /*cuDoubleComplex **/, incx /*int*/);
  // End
}
