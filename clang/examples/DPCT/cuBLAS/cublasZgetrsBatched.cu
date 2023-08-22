#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
          const cuDoubleComplex *const *a, int lda, const int *ipiv,
          cuDoubleComplex *const *b, int ldb, int *info, int group_count) {
  // Start
  cublasZgetrsBatched(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
                      n /*int*/, nrhs /*int*/,
                      a /*const cuDoubleComplex *const **/, lda /*int*/,
                      ipiv /*const int **/, b /*cuDoubleComplex *const **/,
                      ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
