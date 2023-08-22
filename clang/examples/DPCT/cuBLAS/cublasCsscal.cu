#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const float *alpha, cuComplex *x,
          int incx) {
  // Start
  cublasCsscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
               x /*cuComplex **/, incx /*int*/);
  // End
}
