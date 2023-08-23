#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, const float *alpha,
          const float *a, int lda, const float *beta, const float *b, int ldb,
          float *c, int ldc) {
  // Start
  cublasSgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
              alpha /*const float **/, a /*const float **/, lda /*int*/,
              beta /*const float **/, b /*const float **/, ldb /*int*/,
              c /*float **/, ldc /*int*/);
  // End
}
