#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int m, int n,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *x, int incx,
          cuDoubleComplex *c, int ldc) {
  // Start
  cublasZdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              m /*int*/, n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
              x /*const cuDoubleComplex **/, incx /*int*/,
              c /*cuDoubleComplex **/, ldc /*int*/);
  // End
}
