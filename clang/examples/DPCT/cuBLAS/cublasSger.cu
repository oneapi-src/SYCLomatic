#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, const float *alpha,
          const float *x, int incx, const float *y, int incy, float *a,
          int lda) {
  // Start
  cublasSger(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
             alpha /*const float **/, x /*const float **/, incx /*int*/,
             y /*const float **/, incy /*int*/, a /*float **/, lda /*int*/);
  // End
}
