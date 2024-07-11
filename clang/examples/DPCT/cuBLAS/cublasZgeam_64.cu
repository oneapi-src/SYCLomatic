#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n,
          const cuDoubleComplex *alpha, const cuDoubleComplex *a, int64_t lda,
          const cuDoubleComplex *beta, const cuDoubleComplex *b, int64_t ldb,
          cuDoubleComplex *c, int64_t ldc) {
  // Start
  cublasZgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuDoubleComplex **/,
                 a /*const cuDoubleComplex **/, lda /*int64_t*/,
                 beta /*const cuDoubleComplex **/,
                 b /*const cuDoubleComplex **/, ldb /*int64_t*/,
                 c /*cuDoubleComplex **/, ldc /*int64_t*/);
  // End
}
