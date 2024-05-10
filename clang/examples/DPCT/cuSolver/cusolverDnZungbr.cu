#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasSideMode_t left_right, int m, int n,
          int k, cuDoubleComplex *a, int lda, const cuDoubleComplex *tau,
          cuDoubleComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnZungbr(
      handle /*cusolverDnHandle_t*/, left_right /*cublasSideMode_t*/, m /*int*/,
      n /*int*/, k /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
      tau /*const cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
      buffer_size /*int*/, info /*int **/);
  // End
}
