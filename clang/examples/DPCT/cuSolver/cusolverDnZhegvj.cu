#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
          cuDoubleComplex *a, int lda, cuDoubleComplex *b, int ldb, double *w,
          cuDoubleComplex *buffer, int buffer_size, int *info,
          syevjInfo_t params) {
  // Start
  cusolverDnZhegvj(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
                   jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuDoubleComplex **/, lda /*int*/,
                   b /*cuDoubleComplex **/, ldb /*int*/, w /*double **/,
                   buffer /*cuDoubleComplex **/, buffer_size /*int*/,
                   info /*int **/, params /*syevjInfo_t*/);
  // End
}
