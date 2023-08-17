#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const float *alpha, const float *x, int incx, const float *y,
          int incy, float *a, int lda) {
  // Start
  cublasSsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const float **/, x /*const float **/,
              incx /*int*/, y /*const float **/, incy /*int*/, a /*float **/,
              lda /*int*/);
  // End
}
