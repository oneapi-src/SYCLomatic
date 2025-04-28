#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const float *alpha, const float *x, int64_t incx, const float *y,
          int64_t incy, float *a, int64_t lda) {
  // Start
  cublasSsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 n /*int64_t*/, alpha /*const float **/, x /*const float **/,
                 incx /*int64_t*/, y /*const float **/, incy /*int64_t*/,
                 a /*float **/, lda /*int64_t*/);
  // End
}
