#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuComplex *a, int lda,
          cuComplex *buffer, int *ipiv, int *info) {
  // Start
  cusolverDnCgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuComplex **/, lda /*int*/, buffer /*cuComplex **/,
                   ipiv /*int **/, info /*int **/);
  // End
}
