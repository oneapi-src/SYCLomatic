#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const double *alpha,
          const double *a, int lda, const double *b, int ldb,
          const double *beta, double *c, int ldc) {
  // Start
  cublasDgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
              alpha /*const double **/, a /*const double **/, lda /*int*/,
              b /*const double **/, ldb /*int*/, beta /*const double **/,
              c /*double **/, ldc /*int*/);
  // End
}
