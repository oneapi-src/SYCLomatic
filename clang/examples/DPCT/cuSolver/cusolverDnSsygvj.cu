#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *a,
          int lda, float *b, int ldb, float *w, float *buffer, int buffer_size,
          int *info, syevjInfo_t params) {
  // Start
  cusolverDnSsygvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
                   jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*float **/, lda /*int*/, b /*float **/,
                   ldb /*int*/, w /*float **/, buffer /*float **/,
                   buffer_size /*int*/, info /*int **/, params /*syevjInfo_t*/);
  // End
}
