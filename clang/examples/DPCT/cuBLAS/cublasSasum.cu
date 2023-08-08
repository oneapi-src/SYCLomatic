#include "cublas_v2.h"

void test(cublasHandle_t h, int n, const float *x, int incx, float *res) {
  // Start
  cublasSasum(h /*cublasHandle_t*/, n /*int*/, x /*const float **/,
              incx /*int*/, res /*float* */);
  // End
}
