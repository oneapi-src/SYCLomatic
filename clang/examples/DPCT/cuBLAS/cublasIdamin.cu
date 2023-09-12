// Option: --no-dry-pattern
#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *x, int incx, int *res) {
  // Start
  cublasIdamin(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
               incx /*int*/, res /*int **/);
  // End
}
