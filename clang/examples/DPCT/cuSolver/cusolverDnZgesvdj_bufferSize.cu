#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
          int n, const cuDoubleComplex *a, int lda, const double *s,
          const cuDoubleComplex *u, int ldu, const cuDoubleComplex *v, int ldv,
          gesvdjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnZgesvdj_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/, econ /*int*/,
      m /*int*/, n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
      s /*const double **/, u /*const cuDoubleComplex **/, ldu /*int*/,
      v /*const cuDoubleComplex **/, ldv /*int*/, &buffer_size /*int **/,
      params /*gesvdjInfo_t*/);
  // End
}
