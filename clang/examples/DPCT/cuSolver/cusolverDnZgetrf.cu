#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *a, int lda,
          cuDoubleComplex *buffer, int *ipiv, int *info) {
  // Start
  cusolverDnZgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuDoubleComplex **/, lda /*int*/,
                   buffer /*cuDoubleComplex **/, ipiv /*int **/,
                   info /*int **/);
  // End
}
