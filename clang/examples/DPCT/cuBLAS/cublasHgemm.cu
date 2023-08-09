#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const __half *alpha,
          const __half *a, int lda, const __half *b, int ldb,
          const __half *beta, __half *c, int ldc) {
  // Start
  cublasHgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
              alpha /*const __half **/, a /*const __half **/, lda /*int*/,
              b /*const __half **/, ldb /*int*/, beta /*const __half **/,
              c /*__half **/, ldc /*int*/);
  // End
}
