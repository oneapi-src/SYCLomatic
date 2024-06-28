#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n,
          const cuComplex *alpha, const cuComplex *a, int64_t lda,
          const cuComplex *beta, const cuComplex *b, int64_t ldb, cuComplex *c,
          int64_t ldc) {
  // Start
  cublasCgeam_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                 transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                 alpha /*const cuComplex **/, a /*const cuComplex **/,
                 lda /*int64_t*/, beta /*const cuComplex **/,
                 b /*const cuComplex **/, ldb /*int64_t*/, c /*cuComplex **/,
                 ldc /*int64_t*/);
  // End
}
