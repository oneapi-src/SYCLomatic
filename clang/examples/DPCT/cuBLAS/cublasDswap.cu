#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, double *x, int incx, double *y,
          int incy) {
  // Start
  cublasDswap(handle /*cublasHandle_t*/, n /*int*/, x /*double **/,
              incx /*int*/, y /*double **/, incy /*int*/);
  // End
}
