#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const double *alpha, const double *x, int incx, const double *y,
          int incy, double *a, int lda) {
  // Start
  cublasDsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const double **/, x /*const double **/,
              incx /*int*/, y /*const double **/, incy /*int*/, a /*double **/,
              lda /*int*/);
  // End
}
