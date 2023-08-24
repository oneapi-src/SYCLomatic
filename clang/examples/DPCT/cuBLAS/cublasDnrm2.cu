#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *x, int incx,
          double *res) {
  // Start
  cublasDnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
              incx /*int*/, res /*double **/);
  // End
}
