#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const float *alpha, const float *x, int incx, const float *y,
          int incy, float *a) {
  // Start
  cublasSspr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const float **/, x /*const float **/,
              incx /*int*/, y /*const float **/, incy /*int*/, a /*float **/);
  // End
}
