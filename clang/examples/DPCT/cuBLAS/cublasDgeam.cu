#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, const double *alpha,
          const double *a, int lda, const double *beta, const double *b,
          int ldb, double *c, int ldc) {
  // Start
  cublasDgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
              alpha /*const double **/, a /*const double **/, lda /*int*/,
              beta /*const double **/, b /*const double **/, ldb /*int*/,
              c /*double **/, ldc /*int*/);
  // End
}
