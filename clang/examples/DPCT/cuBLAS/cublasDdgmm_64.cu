#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int64_t m,
          int64_t n, const double *a, int64_t lda, const double *x,
          int64_t incx, double *c, int64_t ldc) {
  // Start
  cublasDdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 m /*int64_t*/, n /*int64_t*/, a /*const double **/,
                 lda /*int64_t*/, x /*const double **/, incx /*int64_t*/,
                 c /*double **/, ldc /*int64_t*/);
  // End
}
