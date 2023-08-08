#include "cublas_v2.h"

void test(cublasHandle_t h, int n, const cuComplex *x, int incx, int *res) {
  // Start
  cublasIcamin(h /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
               incx /*int*/, res /*int **/);
  // End
}
