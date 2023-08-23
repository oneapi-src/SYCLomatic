#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *alpha, const double *x,
          int incx, double *y, int incy) {
  // Start
  cublasDaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
              x /*const double **/, incx /*int*/, y /*double **/, incy /*int*/);
  // End
}
