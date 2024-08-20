#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int64_t m,
          int64_t n, const float *a, int64_t lda, const float *x, int64_t incx,
          float *c, int64_t ldc) {
  // Start
  cublasSdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 m /*int64_t*/, n /*int64_t*/, a /*const float **/,
                 lda /*int64_t*/, x /*const float **/, incx /*int64_t*/,
                 c /*float **/, ldc /*int64_t*/);
  // End
}
