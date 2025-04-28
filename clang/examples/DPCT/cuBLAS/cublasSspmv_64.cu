#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const float *alpha, const float *a, const float *x, int64_t incx,
          const float *beta, float *y, int64_t incy) {
  // Start
  cublasSspmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 n /*int64_t*/, alpha /*const float **/, a /*const float **/,
                 x /*const float **/, incx /*int64_t*/, beta /*const float **/,
                 y /*float **/, incy /*int64_t*/);
  // End
}
