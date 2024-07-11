#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const cuDoubleComplex *alpha,
          cuDoubleComplex *x, int64_t incx) {
  // Start
  cublasZscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 alpha /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
                 incx /*int64_t*/);
  // End
}
