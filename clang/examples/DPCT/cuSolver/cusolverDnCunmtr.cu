#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right,
          cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
          cuComplex *a, int lda, cuComplex *tau, cuComplex *c, int ldc,
          cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCunmtr(handle /*cusolverDnHandle_t*/,
                   left_right /*cublasSideMode_t*/, uplo /*cublasFillMode_t*/,
                   trans /*cublasOperation_t*/, m /*int*/, n /*int*/,
                   a /*cuComplex **/, lda /*int*/, tau /*cuComplex **/,
                   c /*cuComplex **/, ldc /*int*/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
