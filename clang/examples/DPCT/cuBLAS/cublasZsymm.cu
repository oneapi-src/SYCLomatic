#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int m, int n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int lda,
          const cuDoubleComplex *b, int ldb, const cuDoubleComplex *beta,
          cuDoubleComplex *c, int ldc) {
  // Start
  cublasZsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
              lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
              beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
              ldc /*int*/);
  // End
}
