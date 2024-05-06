#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int64_t m, int64_t n,
          const cuComplex *alpha, const cuComplex *a, int64_t lda,
          const cuComplex *b, int64_t ldb, const cuComplex *beta, cuComplex *c,
          int64_t ldc) {
  // Start
  cublasChemm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
                 upper_lower /*cublasFillMode_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuComplex **/, a /*const cuComplex **/,
                 lda /*int64_t*/, b /*const cuComplex **/, ldb /*int64_t*/,
                 beta /*const cuComplex **/, c /*cuComplex **/,
                 ldc /*int64_t*/);
  // End
}
