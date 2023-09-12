#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const double *alpha,
          const double *const *a, int lda, const double *const *b, int ldb,
          const double *beta, double *const *c, int ldc, int group_count) {
  // Start
  cublasDgemmBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const double **/, a /*const double *const **/, lda /*int*/,
      b /*const double *const **/, ldb /*int*/, beta /*const double **/,
      c /*double *const **/, ldc /*int*/, group_count /*int*/);
  // End
}
