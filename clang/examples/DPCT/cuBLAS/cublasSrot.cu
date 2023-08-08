#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy,
          const float *c, const float *s) {
  // Start
  cublasSrot(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
             y /*float **/, incy /*int*/, c /*const float **/,
             s /*const float **/);
  // End
}
