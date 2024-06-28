#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, cuComplex *a, int lda, float *s, cuComplex *u, int ldu,
          cuComplex *v, int ldv, cuComplex *buffer, int buffer_size, int *info,
          gesvdjInfo_t params) {
  // Start
  cusolverDnCgesvdj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    econ /*int*/, m /*int*/, n /*int*/, a /*cuComplex **/,
                    lda /*int*/, s /*float **/, u /*cuComplex **/, ldu /*int*/,
                    v /*cuComplex **/, ldv /*int*/, buffer /*cuComplex **/,
                    buffer_size /*int*/, info /*int **/,
                    params /*gesvdjInfo_t*/);
  // End
}
