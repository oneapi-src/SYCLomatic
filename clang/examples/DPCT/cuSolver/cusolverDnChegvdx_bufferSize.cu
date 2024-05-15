#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cusolverEigRange_t range,
          cublasFillMode_t uplo, int n, const cuComplex *a, int lda,
          const cuComplex *b, int ldb, float vl, float vu, int il, int iu,
          int *h_meig, const float *w) {
  // Start
  int buffer_size;
  cusolverDnChegvdx_bufferSize(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
      uplo /*cublasFillMode_t*/, n /*int*/, a /*const cuComplex **/,
      lda /*int*/, b /*const cuComplex **/, ldb /*int*/, vl /*float*/,
      vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
      w /*const float **/, &buffer_size /*int **/);
  // End
}
