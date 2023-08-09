#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const float *alpha,
          const float *a, int lda, long long int stridea, const float *b,
          int ldb, long long int strideb, const float *beta, float *c, int ldc,
          long long int stridec, int group_count) {
  // Start
  cublasSgemmStridedBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const float **/, a /*const float **/, lda /*int*/,
      stridea /*long long int*/, b /*const float **/, ldb /*int*/,
      strideb /*long long int*/, beta /*const float **/, c /*float **/,
      ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
  // End
}
