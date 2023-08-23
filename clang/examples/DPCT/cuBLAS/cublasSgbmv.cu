#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl,
          int ku, const float *alpha, const float *a, int lda, const float *x,
          int incx, const float *beta, float *y, int incy) {
  // Start
  cublasSgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
              n /*int*/, kl /*int*/, ku /*int*/, alpha /*const float **/,
              a /*const float **/, lda /*int*/, x /*const float **/,
              incx /*int*/, beta /*const float **/, y /*float **/,
              incy /*int*/);
  // End
}
