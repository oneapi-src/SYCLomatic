#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex *alpha,
          const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy,
          cuComplex *a, int64_t lda) {
  // Start
  cublasCgeru_64(handle /*cublasHandle_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuComplex **/, x /*const cuComplex **/,
                 incx /*int64_t*/, y /*const cuComplex **/, incy /*int64_t*/,
                 a /*cuComplex **/, lda /*int64_t*/);
  // End
}
