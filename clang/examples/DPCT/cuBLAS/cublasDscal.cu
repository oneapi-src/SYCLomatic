#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *alpha, double *x,
          int incx) {
  // Start
  cublasDscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
              x /*double **/, incx /*int*/);
  // End
}
