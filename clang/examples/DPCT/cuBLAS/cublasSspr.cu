#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const float *alpha, const float *x, int incx, float *a) {
  // Start
  cublasSspr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
             n /*int*/, alpha /*const float **/, x /*const float **/,
             incx /*int*/, a /*float**/);
  // End
}
