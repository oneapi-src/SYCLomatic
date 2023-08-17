#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *const *a,
          int lda, const int *ipiv, cuDoubleComplex *const *c, int ldc,
          int *info, int group_count) {
  // Start
  cublasZgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*const cuDoubleComplex *const **/, lda /*int*/,
                      ipiv /*const int **/, c /*cuDoubleComplex *const **/,
                      ldc /*int*/, info /*int **/, group_count /*int*/);
  // End
}
