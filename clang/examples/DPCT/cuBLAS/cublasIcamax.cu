// Option: --no-dry-pattern
#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *x, int incx,
          int *res) {
  // Start
  cublasIcamax(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
               incx /*int*/, res /*int **/);
  // End
}
