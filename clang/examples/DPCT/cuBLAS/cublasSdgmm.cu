#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int m, int n,
          const float *a, int lda, const float *x, int incx, float *c,
          int ldc) {
  // Start
  cublasSdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              m /*int*/, n /*int*/, a /*const float **/, lda /*int*/,
              x /*const float **/, incx /*int*/, c /*float **/, ldc /*int*/);
  // End
}
