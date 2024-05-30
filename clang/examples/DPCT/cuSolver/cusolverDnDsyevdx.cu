#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *a,
          int lda, double vl, double vu, int il, int iu, int *h_meig, double *w,
          double *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnDsyevdx(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                    range /*cusolverEigRange_t*/, uplo /*cublasFillMode_t*/,
                    n /*int*/, a /*double **/, lda /*int*/, vl /*double*/,
                    vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/,
                    w /*double **/, buffer /*double **/, buffer_size /*int*/,
                    info /*int **/);
  // End
}
