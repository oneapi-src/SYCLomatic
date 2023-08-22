#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
          const cuComplex *a, int lda, const cuComplex *b, int ldb,
          const cuComplex *beta, cuComplex *c, int ldc) {
  // Start
  cublasCgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
              alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
              b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
              c /*cuComplex **/, ldc /*int*/);
  // End
}
