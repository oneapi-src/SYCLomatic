#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, float *const *a, int lda, int *ipiv,
          int *info, int group_count) {
  // Start
  cublasSgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*float *const **/, lda /*int*/, ipiv /*int **/,
                      info /*int **/, group_count /*int*/);
  // End
}
