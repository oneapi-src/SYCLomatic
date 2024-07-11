#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cusolverEigRange_t range,
          cublasFillMode_t uplo, int n, double *a, int lda, double *b, int ldb,
          double vl, double vu, int il, int iu, int *h_meig, double *w,
          double *buffer, int buffer_size, int *info) {
  // Start
  cusolverDnDsygvdx(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
                    jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
                    uplo /*cublasFillMode_t*/, n /*int*/, a /*double **/,
                    lda /*int*/, b /*double **/, ldb /*int*/, vl /*double*/,
                    vu /*double*/, il /*int*/, iu /*int*/, h_meig /*int **/,
                    w /*double **/, buffer /*double **/, buffer_size /*int*/,
                    info /*int **/);
  // End
}
