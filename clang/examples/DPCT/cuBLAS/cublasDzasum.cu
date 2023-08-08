#include "cublas_v2.h"

void test(cublasHandle_t h, int n, const cuDoubleComplex *x, int incx,
          double *res) {
  // Start
  cublasDzasum(h /*cublasHandle_t*/, n /*int*/, x /*const cuDoubleComplex **/,
               incx /*int*/, res /*double* */);
  // End
}
