#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cusolverEigRange_t range, cublasFillMode_t uplo, int n, float *a,
          int lda, float vl, float vu, int il, int iu, int *h_meig, float *w,
          float *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnSsyevdx(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/,
                    n /*int*/, a /*float **/, lda /*int*/, vl /*float*/,
                    vu /*float*/, il /*int*/, iu /*int*/, h_meig /*int **/,
                    w /*float **/, buffer /*float **/, buffer_size /*int*/,
                    info /*int **/);
  // End
}
