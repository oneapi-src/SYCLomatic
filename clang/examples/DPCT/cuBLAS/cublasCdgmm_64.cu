#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int64_t m,
          int64_t n, const cuComplex *a, int64_t lda, const cuComplex *x,
          int64_t incx, cuComplex *c, int64_t ldc) {
  // Start
  cublasCdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 m /*int64_t*/, n /*int64_t*/, a /*const cuComplex **/,
                 lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
                 c /*cuComplex **/, ldc /*int64_t*/);
  // End
}
