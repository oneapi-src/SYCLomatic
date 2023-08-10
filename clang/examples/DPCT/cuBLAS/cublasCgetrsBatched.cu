#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const cuComplex *const *a, int lda, const int *ipiv,
          cuComplex *const *b, int ldb, int *info, int group_count) {
  // Start
  cublasCgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                      n /*int*/, nrhs /*int*/, a /*const cuComplex *const **/,
                      lda /*int*/, ipiv /*const int **/,
                      b /*cuComplex *const **/, ldb /*int*/, info /*int **/,
                      group_count /*int*/);
  // End
}
