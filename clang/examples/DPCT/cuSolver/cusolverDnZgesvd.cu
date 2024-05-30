#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
          int n, cuDoubleComplex *a, int lda, double *s, cuDoubleComplex *u,
          int ldu, cuDoubleComplex *vt, int ldvt, cuDoubleComplex *buffer,
          int buffer_size, double *buffer_for_real, int *info) {
  // Start
  cusolverDnZgesvd(
      handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
      jobvt /*signed char*/, m /*int*/, n /*int*/, a /*cuDoubleComplex **/,
      lda /*int*/, s /*double **/, u /*cuDoubleComplex **/, ldu /*int*/,
      vt /*cuDoubleComplex **/, ldvt /*int*/, buffer /*cuDoubleComplex **/,
      buffer_size /*int*/, buffer_for_real /*double **/, info /*int **/);
  // End
}
