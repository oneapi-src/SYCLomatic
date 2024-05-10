#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, const cuComplex *a, int lda, const cuComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnCungbr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
      tau /*const cuComplex **/, &buffer_size /*int **/);
  // End
}
