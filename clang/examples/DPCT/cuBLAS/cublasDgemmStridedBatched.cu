#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const double *alpha,
          const double *a, int lda, long long int stridea, const double *b,
          int ldb, long long int strideb, const double *beta, double *c,
          int ldc, long long int stridec, int group_count) {
  // Start
  cublasDgemmStridedBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const double **/, a /*const double **/, lda /*int*/,
      stridea /*long long int*/, b /*const double **/, ldb /*int*/,
      strideb /*long long int*/, beta /*const double **/, c /*double **/,
      ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
  // End
}
