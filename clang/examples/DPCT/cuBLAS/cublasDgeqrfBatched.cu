#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, double *const *a, int lda,
          double *const *tau, int *info, int group_count) {
  // Start
  cublasDgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
                      a /*double *const **/, lda /*int*/,
                      tau /*double *const **/, info /*int **/,
                      group_count /*int*/);
  // End
}
