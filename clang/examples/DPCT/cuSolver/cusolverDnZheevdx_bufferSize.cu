#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cusolverEigRange_t range, cublasFillMode_t uplo, int n,
          const cuDoubleComplex *a, int lda, double vl, double vu, int il,
          int iu, int *h_meig, const double *w) {
  // Start
  int buffer_size;
  cusolverDnZheevdx_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
      range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/, vl /*double*/, vu /*double*/,
      il /*int*/, iu /*int*/, h_meig /*int **/, w /*const double **/,
      &buffer_size /*int **/);
  // End
}
