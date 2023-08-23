#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const float *alpha, const float *a, const float *x, int incx,
          const float *beta, float *y, int incy) {
  // Start
  cublasSspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const float **/, a /*const float **/,
              x /*const float **/, incx /*int*/, beta /*const float **/,
              y /*float **/, incy /*int*/);
  // End
}
