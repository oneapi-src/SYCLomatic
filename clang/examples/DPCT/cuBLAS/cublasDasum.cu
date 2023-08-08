#include "cublas_v2.h"

void test(cublasHandle_t h, int n, const double *x, int incx, double *res) {
  // Start
  cublasDasum(h /*cublasHandle_t*/, n /*int*/, x /*const double **/,
              incx /*int*/, res /*double* */);
  // End
}
