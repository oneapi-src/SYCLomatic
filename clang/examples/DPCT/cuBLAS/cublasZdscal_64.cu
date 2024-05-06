#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const double *alpha,
          cuDoubleComplex *x, int64_t incx) {
  // Start
  cublasZdscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                  alpha /*const double **/, x /*cuDoubleComplex **/,
                  incx /*int64_t*/);
  // End
}
