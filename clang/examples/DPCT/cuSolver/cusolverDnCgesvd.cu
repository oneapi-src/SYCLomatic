#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
          int n, cuComplex *a, int lda, float *s, cuComplex *u, int ldu,
          cuComplex *vt, int ldvt, cuComplex *buffer, int buffer_size,
          float *buffer_for_real, int *info) {
  // Start
  cusolverDnCgesvd(handle /*cusolverDnHandle_t*/, jobu /*signed char*/,
                   jobvt /*signed char*/, m /*int*/, n /*int*/,
                   a /*cuComplex **/, lda /*int*/, s /*float **/,
                   u /*cuComplex **/, ldu /*int*/, vt /*cuComplex **/,
                   ldvt /*int*/, buffer /*cuComplex **/, buffer_size /*int*/,
                   buffer_for_real /*float **/, info /*int **/);
  // End
}
