#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, double *a, int lda,
          double *d, double *e, double *tau_q, double *tau_p, double *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnDgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*double **/, lda /*int*/, d /*double **/, e /*double **/,
                   tau_q /*double **/, tau_p /*double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
