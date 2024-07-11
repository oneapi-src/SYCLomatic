#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int64_t m, int64_t n,
          const float *alpha, const float *a, int64_t lda, const float *b,
          int64_t ldb, const float *beta, float *c, int64_t ldc) {
  // Start
  cublasSsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
                 b /*const float **/, ldb /*int64_t*/, beta /*const float **/,
                 c /*float **/, ldc /*int64_t*/);
  // End
}
