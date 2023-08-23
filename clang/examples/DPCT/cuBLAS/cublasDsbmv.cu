#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n, int k,
          const double *alpha, const double *a, int lda, const double *x,
          int incx, const double *beta, double *y, int incy) {
  // Start
  cublasDsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, k /*int*/, alpha /*const double **/,
              a /*const double **/, lda /*int*/, x /*const double **/,
              incx /*int*/, beta /*const double **/, y /*double **/,
              incy /*int*/);
  // End
}
