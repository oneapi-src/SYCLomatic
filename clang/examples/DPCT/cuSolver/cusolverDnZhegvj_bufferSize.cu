#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *b, int ldb,
          const double *w, syevjInfo_t params) {
  // Start
  int buffer_size;
  cusolverDnZhegvj_bufferSize(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuDoubleComplex **/, lda /*int*/, b /*const cuDoubleComplex **/,
      ldb /*int*/, w /*const double **/, &buffer_size /*int **/,
      params /*syevjInfo_t*/);
  // End
}
