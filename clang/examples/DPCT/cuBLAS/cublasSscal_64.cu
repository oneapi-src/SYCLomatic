#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const float *alpha, float *x,
          int64_t incx) {
  // Start
  cublasSscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 alpha /*const float **/, x /*float **/, incx /*int64_t*/);
  // End
}
