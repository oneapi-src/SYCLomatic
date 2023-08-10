#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, cuDoubleComplex *const *a,
          int lda, cuDoubleComplex *const *tau, int *info, int group_count) {
  // Start
  cublasZgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
                      a /*cuDoubleComplex *const **/, lda /*int*/,
                      tau /*cuDoubleComplex *const **/, info /*int **/,
                      group_count /*int*/);
  // End
}
