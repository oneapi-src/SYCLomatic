#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuComplex *a, int lda,
          cuComplex *tau, cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuComplex **/, lda /*int*/, tau /*cuComplex **/,
                   buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
