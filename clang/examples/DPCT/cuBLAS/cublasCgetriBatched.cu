#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuComplex *const *a, int lda,
          const int *ipiv, cuComplex *const *c, int ldc, int *info,
          int group_count) {
  // Start
  cublasCgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*const cuComplex *const **/, lda /*int*/,
                      ipiv /*const int **/, c /*cuComplex *const **/,
                      ldc /*int*/, info /*int **/, group_count /*int*/);
  // End
}
