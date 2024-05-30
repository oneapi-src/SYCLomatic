#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cusolverEigRange_t range, cublasFillMode_t uplo, int n,
          const float *a, int lda, float vl, float vu, int il, int iu,
          int *h_meig, const float *w) {
  // Start
  int buffer_size;
  cusolverDnSsyevdx_bufferSize(
      handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
      range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const float **/, lda /*int*/, vl /*float*/, vu /*float*/, il /*int*/,
      iu /*int*/, h_meig /*int **/, w /*const float **/,
      &buffer_size /*int **/);
  // End
}
