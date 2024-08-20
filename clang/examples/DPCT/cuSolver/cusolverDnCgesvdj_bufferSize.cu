#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, const cuComplex *a, int lda, const float *s,
          const cuComplex *u, int ldu, const cuComplex *v, int ldv,
          gesvdjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnCgesvdj_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
      m /*int*/, n /*int*/, a /*const cuComplex **/, lda /*int*/,
      s /*const float **/, u /*const cuComplex **/, ldu /*int*/,
      v /*const cuComplex **/, ldv /*int*/, &buffer_size /*int **/,
      params /*gesvdjInfo_t*/);
  // End
}
