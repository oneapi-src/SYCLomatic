#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const double *alpha, const double *a, const double *x, int incx,
          const double *beta, double *y, int incy) {
  // Start
  cublasDspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const double **/, a /*const double **/,
              x /*const double **/, incx /*int*/, beta /*const double **/,
              y /*double **/, incy /*int*/);
  // End
}
