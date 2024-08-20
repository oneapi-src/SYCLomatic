#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const float *alpha, const float *x,
          int64_t incx, float *y, int64_t incy) {
  // Start
  cublasSaxpy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 alpha /*const float **/, x /*const float **/, incx /*int64_t*/,
                 y /*float **/, incy /*int64_t*/);
  // End
}
