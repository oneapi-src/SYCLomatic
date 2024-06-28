#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int64_t m,
          int64_t n, const cuDoubleComplex *a, int64_t lda,
          const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *c,
          int64_t ldc) {
  // Start
  cublasZdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 m /*int64_t*/, n /*int64_t*/, a /*const cuDoubleComplex **/,
                 lda /*int64_t*/, x /*const cuDoubleComplex **/,
                 incx /*int64_t*/, c /*cuDoubleComplex **/, ldc /*int64_t*/);
  // End
}
