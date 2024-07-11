#include "cublas_v2.h"

void test(int64_t n, int64_t elementsize, const void *x, int64_t incx, void *y,
          int64_t incy) {
  // Start
  cublasGetVector_64(n /*int64_t*/, elementsize /*int64_t*/, x /*const void **/,
                     incx /*int64_t*/, y /*void **/, incy /*int64_t*/);
  // End
}
