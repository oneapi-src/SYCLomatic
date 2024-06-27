#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *a, int lda,
          cuDoubleComplex *tau, cuDoubleComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnZgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuDoubleComplex **/, lda /*int*/,
                   tau /*cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
