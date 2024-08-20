#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, float *a, int lda, float *d,
          float *e, float *tau_q, float *tau_p, float *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnSgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*float **/, lda /*int*/, d /*float **/, e /*float **/,
                   tau_q /*float **/, tau_p /*float **/, buffer /*float **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
