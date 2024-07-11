#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, float *x, int64_t incx, float *y,
          int64_t incy, const float *param) {
  // Start
  cublasSrotm_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*float **/,
                 incx /*int64_t*/, y /*float **/, incy /*int64_t*/,
                 param /*const float **/);
  // End
}
