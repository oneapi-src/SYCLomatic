#include "cublas_v2.h"

void test(cublasHandle_t h, int n, const float *x, int incx, int *res) {
  // Start
  cublasIsamin(h /*cublasHandle_t*/, n /*int*/, x /*const float **/,
               incx /*int*/, res /*int **/);
  // End
}
