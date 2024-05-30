#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, int k, double *a, int lda,
          const double *tau, double *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnDorgqr(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   k /*int*/, a /*double **/, lda /*int*/,
                   tau /*const double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
