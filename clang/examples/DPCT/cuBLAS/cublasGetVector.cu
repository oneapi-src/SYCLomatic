#include "cublas_v2.h"

void test(int n, int elementsize, const void *x, int incx, void *y, int incy) {
  // Start
  cublasGetVector(n /*int*/, elementsize /*int*/, x /*const void **/,
                  incx /*int*/, y /*void **/, incy /*int*/);
  // End
}
