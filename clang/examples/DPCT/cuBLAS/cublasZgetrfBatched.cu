#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, cuDoubleComplex *const *a, int lda,
          int *ipiv, int *info, int group_count) {
  // Start
  cublasZgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*cuDoubleComplex *const **/, lda /*int*/,
                      ipiv /*int **/, info /*int **/, group_count /*int*/);
  // End
}
