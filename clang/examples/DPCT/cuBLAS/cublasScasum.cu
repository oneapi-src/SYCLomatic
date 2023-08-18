#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *x, int incx,
          float *res) {
  // Start
  cublasScasum(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
               incx /*int*/, res /*float **/);
  // End
}
