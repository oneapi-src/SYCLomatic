#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuComplex *a, int lda,
          float *d, float *e, cuComplex *tau_q, cuComplex *tau_p,
          cuComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnCgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuComplex **/, lda /*int*/, d /*float **/, e /*float **/,
                   tau_q /*cuComplex **/, tau_p /*cuComplex **/,
                   buffer /*cuComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
