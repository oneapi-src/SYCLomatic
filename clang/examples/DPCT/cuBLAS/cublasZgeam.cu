#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *beta,
          const cuDoubleComplex *b, int ldb, cuDoubleComplex *c, int ldc) {
  // Start
  cublasZgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
              transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
              alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
              lda /*int*/, beta /*const cuDoubleComplex **/,
              b /*const cuDoubleComplex **/, ldb /*int*/,
              c /*cuDoubleComplex **/, ldc /*int*/);
  // End
}
