#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, double *a, int lda, double *w,
          double *buffer, int buffer_size, int *info, syevjInfo_t params) {
  // Start
  cusolverDnDsyevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                   uplo /*cublasFillMode_t*/, n /*int*/, a /*double **/,
                   lda /*int*/, w /*double **/, buffer /*double **/,
                   buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
  // End
}
