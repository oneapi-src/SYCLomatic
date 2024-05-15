#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
          int n, float *a, int lda, float *s, float *u, int ldu, float *vt,
          int ldvt, float *buffer, int buffer_size, float *buffer_for_real,
          int *info) {
  // Start
  cusolverDnSgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
                   jobvt /*signed char*/, m /*int*/, n /*int*/, a /*float **/,
                   lda /*int*/, s /*float **/, u /*float **/, ldu /*int*/,
                   vt /*float **/, ldvt /*int*/, buffer /*float **/,
                   buffer_size /*int*/, buffer_for_real /*float **/,
                   info /*int **/);
  // End
}
