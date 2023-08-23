#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, const double *alpha,
          const double *x, int incx, const double *y, int incy, double *a,
          int lda) {
  // Start
  cublasDger(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
             alpha /*const double **/, x /*const double **/, incx /*int*/,
             y /*const double **/, incy /*int*/, a /*double **/, lda /*int*/);
  // End
}
