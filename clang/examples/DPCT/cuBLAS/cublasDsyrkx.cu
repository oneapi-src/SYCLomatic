#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int n, int k, const double *alpha,
          const double *a, int lda, const double *b, int ldb,
          const double *beta, double *c, int ldc) {
  // Start
  cublasDsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
               alpha /*const double **/, a /*const double **/, lda /*int*/,
               b /*const double **/, ldb /*int*/, beta /*const double **/,
               c /*double **/, ldc /*int*/);
  // End
}
