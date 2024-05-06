#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int64_t n, int64_t k, const double *alpha,
          const double *a, int64_t lda, const double *b, int64_t ldb,
          const double *beta, double *c, int64_t ldc) {
  // Start
  cublasDsyr2k_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
                  alpha /*const double **/, a /*const double **/,
                  lda /*int64_t*/, b /*const double **/, ldb /*int64_t*/,
                  beta /*const double **/, c /*double **/, ldc /*int64_t*/);
  // End
}
