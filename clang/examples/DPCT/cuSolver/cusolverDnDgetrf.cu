#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, double *a, int lda,
          double *buffer, int *ipiv, int *info) {
  // Start
  cusolverDnDgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*double **/, lda /*int*/, buffer /*double **/,
                   ipiv /*int **/, info /*int **/);
  // End
}
