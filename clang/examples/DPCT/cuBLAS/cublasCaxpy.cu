#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *alpha,
          const cuComplex *x, int incx, cuComplex *y, int incy) {
  // Start
  cublasCaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
              x /*const cuComplex **/, incx /*int*/, y /*cuComplex **/,
              incy /*int*/);
  // End
}
