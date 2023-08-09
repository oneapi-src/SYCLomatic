#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const double *alpha, const double *x, int incx, double *a, int lda) {
  // Start
  cublasDsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
             n /*int*/, alpha /*const double **/, x /*const double **/,
             incx /*int*/, a /*double **/, lda /*int*/);
  // End
}
