#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, const double *a, int lda, const double *s, const double *u,
          int ldu, const double *v, int ldv, gesvdjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnDgesvdj_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
      m /*int*/, n /*int*/, a /*const double **/, lda /*int*/,
      s /*const double **/, u /*const double **/, ldu /*int*/,
      v /*const double **/, ldv /*int*/, &buffer_size /*int **/,
      params /*gesvdjInfo_t*/);
  // End
}
