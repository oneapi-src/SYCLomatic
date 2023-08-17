#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n, int k,
          const float *alpha, const float *a, int lda, const float *x, int incx,
          const float *beta, float *y, int incy) {
  // Start
  cublasSsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, k /*int*/, alpha /*const float **/,
              a /*const float **/, lda /*int*/, x /*const float **/,
              incx /*int*/, beta /*const float **/, y /*float **/,
              incy /*int*/);
  // End
}
