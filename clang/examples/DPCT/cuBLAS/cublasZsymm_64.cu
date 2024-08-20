#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int64_t m, int64_t n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int64_t lda,
          const cuDoubleComplex *b, int64_t ldb, const cuDoubleComplex *beta,
          cuDoubleComplex *c, int64_t ldc) {
  // Start
  cublasZsymm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuDoubleComplex **/,
                 a /*const cuDoubleComplex **/, lda /*int64_t*/,
                 b /*const cuDoubleComplex **/, ldb /*int64_t*/,
                 beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
                 ldc /*int64_t*/);
  // End
}
