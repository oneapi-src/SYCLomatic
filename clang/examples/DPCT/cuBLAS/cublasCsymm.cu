#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int m, int n, const cuComplex *alpha,
          const cuComplex *a, int lda, const cuComplex *b, int ldb,
          const cuComplex *beta, cuComplex *c, int ldc) {
  // Start
  cublasCsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
              alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
              b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
              c /*cuComplex **/, ldc /*int*/);
  // End
}
