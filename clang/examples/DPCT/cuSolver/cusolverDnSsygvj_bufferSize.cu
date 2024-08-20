#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *a,
          int lda, const float *b, int ldb, const float *w,
          syevjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnSsygvj_bufferSize(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const float **/, lda /*int*/, b /*const float **/, ldb /*int*/,
      w /*const float **/, &buffer_size /*int **/, params /*syevjInfo_t*/);
  // End
}
