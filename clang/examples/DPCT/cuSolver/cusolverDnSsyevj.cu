#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, float *a, int lda, float *w,
          float *buffer, int buffer_size, int *info, syevjInfo_t params) {
  // Start
  cusolverDnSsyevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                   uplo /*cublasFillMode_t*/, n /*int*/, a /*float **/,
                   lda /*int*/, w /*float **/, buffer /*float **/,
                   buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
  // End
}
