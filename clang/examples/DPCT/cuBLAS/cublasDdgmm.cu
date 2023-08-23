#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right, int m, int n,
          const double *a, int lda, const double *x, int incx, double *c,
          int ldc) {
  // Start
  cublasDdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              m /*int*/, n /*int*/, a /*const double **/, lda /*int*/,
              x /*const double **/, incx /*int*/, c /*double **/, ldc /*int*/);
  // End
}
