#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, cuComplex *a, int lda,
          const cuComplex *tau, cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCungqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   k /*int*/, a /*cuComplex **/, lda /*int*/,
                   tau /*const cuComplex **/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
