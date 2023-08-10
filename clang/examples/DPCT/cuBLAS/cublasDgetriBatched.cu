#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const double *const *a, int lda,
          const int *ipiv, double *const *c, int ldc, int *info,
          int group_count) {
  // Start
  cublasDgetriBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*const double *const **/, lda /*int*/,
                      ipiv /*const int **/, c /*double *const **/, ldc /*int*/,
                      info /*int **/, group_count /*int*/);
  // End
}
