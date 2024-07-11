#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, const double *alpha,
          const double *a, int64_t lda, const double *beta, const double *b,
          int64_t ldb, double *c, int64_t ldc) {
  // Start
  cublasDgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const double **/, a /*const double **/,
                 lda /*int64_t*/, beta /*const double **/, b /*const double **/,
                 ldb /*int64_t*/, c /*double **/, ldc /*int64_t*/);
  // End
}
