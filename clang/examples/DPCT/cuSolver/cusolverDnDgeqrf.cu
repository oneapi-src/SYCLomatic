#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, double *a, int lda,
          double *tau, double *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnDgeqrf(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*double **/, lda /*int*/, tau /*double **/,
                   buffer /*double **/, buffer_size /*int*/, info /*int **/);
  // End
}
