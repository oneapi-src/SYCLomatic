#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
          int n, double *a, int lda, double *s, double *u, int ldu, double *vt,
          int ldvt, double *buffer, int buffer_size, double *buffer_for_real,
          int *info) {
  // Start
  cusolverDnDgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
                   jobvt /*signed char*/, m /*int*/, n /*int*/, a /*double **/,
                   lda /*int*/, s /*double **/, u /*double **/, ldu /*int*/,
                   vt /*double **/, ldvt /*int*/, buffer /*double **/,
                   buffer_size /*int*/, buffer_for_real /*double **/,
                   info /*int **/);
  // End
}
