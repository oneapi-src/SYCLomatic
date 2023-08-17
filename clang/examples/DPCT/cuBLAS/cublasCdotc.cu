#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *x, int incx,
          const cuComplex *y, int incy, cuComplex *res) {
  // Start
  cublasCdotc(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
              incx /*int*/, y /*const cuComplex **/, incy /*int*/,
              res /*cuComplex **/);
  // End
}
