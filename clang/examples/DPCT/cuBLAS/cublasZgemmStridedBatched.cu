#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int lda,
          long long int stridea, const cuDoubleComplex *b, int ldb,
          long long int strideb, const cuDoubleComplex *beta,
          cuDoubleComplex *c, int ldc, long long int stridec, int group_count) {
  // Start
  cublasZgemmStridedBatched(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
      lda /*int*/, stridea /*long long int*/, b /*const cuDoubleComplex **/,
      ldb /*int*/, strideb /*long long int*/, beta /*const cuDoubleComplex **/,
      c /*cuDoubleComplex **/, ldc /*int*/, stridec /*long long int*/,
      group_count /*int*/);
  // End
}
