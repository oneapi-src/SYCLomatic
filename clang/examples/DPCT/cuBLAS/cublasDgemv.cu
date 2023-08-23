#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
          const double *alpha, const double *a, int lda, const double *x,
          int incx, const double *beta, double *y, int incy) {
  // Start
  cublasDgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
              n /*int*/, alpha /*const double **/, a /*const double **/,
              lda /*int*/, x /*const double **/, incx /*int*/,
              beta /*const double **/, y /*double **/, incy /*int*/);
  // End
}
