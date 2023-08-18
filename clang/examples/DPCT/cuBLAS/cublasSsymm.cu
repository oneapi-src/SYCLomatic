#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int m, int n, const float *alpha,
          const float *a, int lda, const float *b, int ldb, const float *beta,
          float *c, int ldc) {
  // Start
  cublasSsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
              alpha /*const float **/, a /*const float **/, lda /*int*/,
              b /*const float **/, ldb /*int*/, beta /*const float **/,
              c /*float **/, ldc /*int*/);
  // End
}
