#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *x, int incx, double *y,
          int incy) {
  // Start
  cublasDcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
              incx /*int*/, y /*double **/, incy /*int*/);
  // End
}
