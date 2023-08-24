#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, double *const *a, int lda, int *ipiv,
          int *info, int group_count) {
  // Start
  cublasDgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*double *const **/, lda /*int*/, ipiv /*int **/,
                      info /*int **/, group_count /*int*/);
  // End
}
