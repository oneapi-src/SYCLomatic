#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower, int n,
          const cuComplex *alpha, const cuComplex *x, int incx,
          const cuComplex *y, int incy, cuComplex *a) {
  // Start
  cublasChpr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
              incx /*int*/, y /*const cuComplex **/, incy /*int*/,
              a /*cuComplex **/);
  // End
}
