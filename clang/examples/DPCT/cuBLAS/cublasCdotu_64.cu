#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx,
          const cuComplex *y, int64_t incy, cuComplex *res) {
  // Start
  cublasCdotu_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 x /*const cuComplex **/, incx /*int64_t*/,
                 y /*const cuComplex **/, incy /*int64_t*/,
                 res /*cuComplex **/);
  // End
}
