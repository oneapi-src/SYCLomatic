#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, const cuDoubleComplex *a, int lda,
          const cuDoubleComplex *tau) {
  // Start
  int buffer_size;
  cusolverDnZungbr_bufferSize(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
      tau /*const cuDoubleComplex **/, &buffer_size /*int **/);
  // End
}
