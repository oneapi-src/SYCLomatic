#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, cuComplex *const *a, int lda,
          cuComplex *const *tau, int *info, int group_count) {
  // Start
  cublasCgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
                      a /*cuComplex *const **/, lda /*int*/,
                      tau /*cuComplex *const **/, info /*int **/,
                      group_count /*int*/);
  // End
}
