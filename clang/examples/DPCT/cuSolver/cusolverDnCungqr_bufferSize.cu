#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex *a,
          int lda, const cuComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnCungqr_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, k /*int*/, a /*const cuComplex **/,
                              lda /*int*/, tau /*const cuComplex **/,
                              &buffer_size /*int **/);
  // End
}
