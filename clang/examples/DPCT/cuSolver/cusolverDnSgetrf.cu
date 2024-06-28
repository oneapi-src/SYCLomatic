#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, float *a, int lda,
          float *buffer, int *ipiv, int *info) {
  // Start
  cusolverDnSgetrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*float **/, lda /*int*/, buffer /*float **/,
                   ipiv /*int **/, info /*int **/);
  // End
}
