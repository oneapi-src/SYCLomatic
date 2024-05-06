#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int64_t n, int64_t k, const double *alpha,
          const cuDoubleComplex *a, int64_t lda, const double *beta,
          cuDoubleComplex *c, int64_t ldc) {
  // Start
  cublasZherk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                 trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
                 alpha /*const double **/, a /*const cuDoubleComplex **/,
                 lda /*int64_t*/, beta /*const double **/,
                 c /*cuDoubleComplex **/, ldc /*int64_t*/);
  // End
}
