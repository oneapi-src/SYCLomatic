#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const float *alpha, const float *x,
          int incx, float *y, int incy) {
  // Start
  cublasSaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
              x /*const float **/, incx /*int*/, y /*float **/, incy /*int*/);
  // End
}
