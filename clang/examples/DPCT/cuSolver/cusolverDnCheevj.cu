#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
          cublasFillMode_t uplo, int n, cuComplex *a, int lda, float *w,
          cuComplex *buffer, int buffer_size, int *info, syevjInfo_t params) {
  // Start
  cusolverDnCheevj(handle /*cusolverDnHandle_t*/, jobz /*cusolverEigMode_t*/,
                   uplo /*cublasFillMode_t*/, n /*int*/, a /*cuComplex **/,
                   lda /*int*/, w /*float **/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
  // End
}
