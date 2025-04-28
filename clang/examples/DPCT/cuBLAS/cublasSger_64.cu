#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t m, int64_t n, const float *alpha,
          const float *x, int64_t incx, const float *y, int64_t incy, float *a,
          int64_t lda) {
  // Start
  cublasSger_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
                alpha /*const float **/, x /*const float **/, incx /*int64_t*/,
                y /*const float **/, incy /*int64_t*/, a /*float **/,
                lda /*int64_t*/);
  // End
}
