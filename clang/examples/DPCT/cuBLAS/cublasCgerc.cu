#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, const cuComplex *alpha,
          const cuComplex *x, int incx, const cuComplex *y, int incy,
          cuComplex *a, int lda) {
  // Start
  cublasCgerc(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
              alpha /*const cuComplex **/, x /*const cuComplex **/,
              incx /*int*/, y /*const cuComplex **/, incy /*int*/,
              a /*cuComplex **/, lda /*int*/);
  // End
}
