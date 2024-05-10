#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex *a,
          int lda, const cuDoubleComplex *tau, cuDoubleComplex *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnZungqr(
      handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*cuDoubleComplex **/, lda /*int*/, tau /*const cuDoubleComplex **/,
      buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
