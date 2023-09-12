#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, const cuComplex *alpha,
          const cuComplex *a, int lda, const cuComplex *beta,
          const cuComplex *b, int ldb, cuComplex *c, int ldc) {
  // Start
  cublasCgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
              alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
              beta /*const cuComplex **/, b /*const cuComplex **/, ldb /*int*/,
              c /*cuComplex **/, ldc /*int*/);
  // End
}
