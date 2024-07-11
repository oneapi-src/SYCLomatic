#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const double *x, int64_t incx,
          double *res) {
  // Start
  cublasDnrm2_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const double **/,
                 incx /*int64_t*/, res /*double **/);
  // End
}
