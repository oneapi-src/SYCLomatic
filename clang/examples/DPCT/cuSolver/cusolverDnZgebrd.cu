#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *a, int lda,
          double *d, double *e, cuDoubleComplex *tau_q, cuDoubleComplex *tau_p,
          cuDoubleComplex *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnZgebrd(handle /*cusolverDnHandle_t*/, m /*int*/, n /*int*/,
                   a /*cuDoubleComplex **/, lda /*int*/, d /*double **/,
                   e /*double **/, tau_q /*cuDoubleComplex **/,
                   tau_p /*cuDoubleComplex **/, buffer /*cuDoubleComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
