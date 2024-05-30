#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnZungqr_bufferSize(
      handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/,
      tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
  // End
}
