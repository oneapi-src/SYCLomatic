#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const float *alpha,
          const float *a, int lda, const float *b, int ldb, const float *beta,
          float *c, int ldc) {
  // Start
  cublasSgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
              alpha /*const float **/, a /*const float **/, lda /*int*/,
              b /*const float **/, ldb /*int*/, beta /*const float **/,
              c /*float **/, ldc /*int*/);
  // End
}
