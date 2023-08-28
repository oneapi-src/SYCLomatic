#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t upper_lower, int m, int n, const double *alpha,
          const double *a, int lda, const double *b, int ldb,
          const double *beta, double *c, int ldc) {
  // Start
  cublasDsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
              upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
              alpha /*const double **/, a /*const double **/, lda /*int*/,
              b /*const double **/, ldb /*int*/, beta /*const double **/,
              c /*double **/, ldc /*int*/);
  // End
}
