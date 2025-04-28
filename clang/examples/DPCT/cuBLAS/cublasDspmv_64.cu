#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const double *alpha, const double *a, const double *x, int64_t incx,
          const double *beta, double *y, int64_t incy) {
  // Start
  cublasDspmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 n /*int64_t*/, alpha /*const double **/, a /*const double **/,
                 x /*const double **/, incx /*int64_t*/,
                 beta /*const double **/, y /*double **/, incy /*int64_t*/);
  // End
}
