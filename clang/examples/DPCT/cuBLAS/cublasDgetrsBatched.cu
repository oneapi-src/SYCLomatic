#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const double *const *a, int lda, const int *ipiv, double *const *b,
          int ldb, int *info, int group_count) {
  // Start
  cublasDgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                      n /*int*/, nrhs /*int*/, a /*const double *const **/,
                      lda /*int*/, ipiv /*const int **/, b /*double *const **/,
                      ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
