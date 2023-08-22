#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const float *x, int incx,
          const float *y, int incy, float *res) {
  // Start
  cublasSdot(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
             incx /*int*/, y /*const float **/, incy /*int*/, res /*float **/);
  // End
}
