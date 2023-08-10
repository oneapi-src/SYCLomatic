#include "cublas_v2.h"

void test(cublasHandle_t handle, int m, int n, float *const *a, int lda,
          float *const *tau, int *info, int group_count) {
  // Start
  cublasSgeqrfBatched(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
                      a /*float *const **/, lda /*int*/, tau /*float *const **/,
                      info /*int **/, group_count /*int*/);
  // End
}
