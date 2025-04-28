#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int64_t n,
          const double *alpha, const double *x, int64_t incx, const double *y,
          int64_t incy, double *a, int64_t lda) {
  // Start
  cublasDsyr2_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 n /*int64_t*/, alpha /*const double **/, x /*const double **/,
                 incx /*int64_t*/, y /*const double **/, incy /*int64_t*/,
                 a /*double **/, lda /*int64_t*/);
  // End
}
