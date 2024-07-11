#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const cuComplex *alpha,
          cuComplex *x, int64_t incx) {
  // Start
  cublasCscal_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 alpha /*const cuComplex **/, x /*cuComplex **/,
                 incx /*int64_t*/);
  // End
}
