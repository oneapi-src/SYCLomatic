#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cusolverEigRange_t range,
          cublasFillMode_t uplo, int n, cuDoubleComplex *a, int lda,
          cuDoubleComplex *b, int ldb, double vl, double vu, int il, int iu,
          int *h_meig, double *w, cuDoubleComplex *buffer, int buffer_size,
          int *info) {
  // Start
  cusolverDnZhegvdx(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
      uplo /*cublasFillMode_t*/, n /*int*/, a /*cuDoubleComplex **/,
      lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/, vl /*double*/,
      vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/, w /*double **/,
      buffer /*cuDoubleComplex **/, buffer_size /*int*/, info /*int **/);
  // End
}
