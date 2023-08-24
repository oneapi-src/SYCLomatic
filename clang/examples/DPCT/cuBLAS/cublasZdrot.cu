#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
          cuDoubleComplex *y, int incy, const double *c, const double *s) {
  // Start
  cublasZdrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
              incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/,
              c /*const double **/, s /*const double **/);
  // End
}
