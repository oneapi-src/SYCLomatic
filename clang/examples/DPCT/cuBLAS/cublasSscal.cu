#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const float *alpha, float *x,
          int incx) {
  // Start
  cublasSscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
              x /*float **/, incx /*int*/);
  // End
}
