#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *x, int incx,
          cuComplex *y, int incy) {
  // Start
  cublasCcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
              incx /*int*/, y /*cuComplex **/, incy /*int*/);
  // End
}
