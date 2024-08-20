#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const float *x, int64_t incx,
          const float *y, int64_t incy, float *res) {
  // Start
  cublasSdot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
                incx /*int64_t*/, y /*const float **/, incy /*int64_t*/,
                res /*float **/);
  // End
}
