#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int64_t n, int64_t k, const float *alpha,
          const cuComplex *a, int64_t lda, const float *beta, cuComplex *c,
          int64_t ldc) {
  // Start
  cublasCherk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
                 alpha /*const float **/, a /*const cuComplex **/,
                 lda /*int64_t*/, beta /*const float **/, c /*cuComplex **/,
                 ldc /*int64_t*/);
  // End
}
