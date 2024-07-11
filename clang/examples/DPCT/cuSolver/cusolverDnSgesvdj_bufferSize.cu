#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, const float *a, int lda, const float *s, const float *u,
          int ldu, const float *v, int ldv, gesvdjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnSgesvdj_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
      m /*int*/, n /*int*/, a /*const float **/, lda /*int*/,
      s /*const float **/, u /*const float **/, ldu /*int*/,
      v /*const float **/, ldv /*int*/, &buffer_size /*int **/,
      params /*gesvdjInfo_t*/);
  // End
}
