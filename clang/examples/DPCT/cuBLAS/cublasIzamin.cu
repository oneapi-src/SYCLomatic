// Option: --no-dry-pattern
#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
          int *res) {
  // Start
  cublasIzamin(handle /*cublasHandle_t*/, n /*int*/,
               x /*const cuDoubleComplex **/, incx /*int*/, res /*int **/);
  // End
}
