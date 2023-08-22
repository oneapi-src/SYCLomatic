#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
          const cuComplex *a, int lda, long long int stridea,
          const cuComplex *b, int ldb, long long int strideb,
          const cuComplex *beta, cuComplex *c, int ldc, long long int stridec,
          int group_count) {
  // Start
  cublasCgemmStridedBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
      stridea /*long long int*/, b /*const cuComplex **/, ldb /*int*/,
      strideb /*long long int*/, beta /*const cuComplex **/, c /*cuComplex **/,
      ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
  // End
}
