#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
          int64_t kl, int64_t ku, const float *alpha, const float *a,
          int64_t lda, const float *x, int64_t incx, const float *beta,
          float *y, int64_t incy) {
  // Start
  cublasSgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                 m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
                 alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
                 x /*const float **/, incx /*int64_t*/, beta /*const float **/,
                 y /*float **/, incy /*int64_t*/);
  // End
}
