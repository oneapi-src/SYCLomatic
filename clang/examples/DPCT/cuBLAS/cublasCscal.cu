#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x,
          int incx) {
  // Start
  cublasCscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
              x /*cuComplex **/, incx /*int*/);
  // End
}
