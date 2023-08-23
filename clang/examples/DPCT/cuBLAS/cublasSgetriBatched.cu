#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const float *const *a, int lda,
          const int *ipiv, float *const *c, int ldc, int *info,
          int group_count) {
  // Start
  cublasSgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*const float *const **/, lda /*int*/,
                      ipiv /*const int **/, c /*float *const **/, ldc /*int*/,
                      info /*int **/, group_count /*int*/);
  // End
}
