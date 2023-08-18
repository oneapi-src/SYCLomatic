#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int m, int n,
          const cuComplex *a, int lda, const cuComplex *x, int incx,
          cuComplex *c, int ldc) {
  // Start
  cublasCdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              m /*int*/, n /*int*/, a /*const cuComplex **/, lda /*int*/,
              x /*const cuComplex **/, incx /*int*/, c /*cuComplex **/,
              ldc /*int*/);
  // End
}
