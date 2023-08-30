#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int n, int k, const cuComplex *alpha,
          const cuComplex *a, int lda, const cuComplex *b, int ldb,
          const cuComplex *beta, cuComplex *c, int ldc) {
  // Start
  cublasCsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
               alpha /*const cuComplex **/, a /*const cuComplex **/,
               lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
               beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
  // End
}
