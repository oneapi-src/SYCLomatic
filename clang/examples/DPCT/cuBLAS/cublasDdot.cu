#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *x, int incx,
          const double *y, int incy, double *res) {
  // Start
  cublasDdot(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
             incx /*int*/, y /*const double **/, incy /*int*/,
             res /*double **/);
  // End
}
