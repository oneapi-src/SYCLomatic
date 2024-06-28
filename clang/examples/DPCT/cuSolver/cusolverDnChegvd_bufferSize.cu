#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
          const cuComplex *a, int lda, const cuComplex *b, int ldb,
          const float *w) {
  // Start
  int buffer_size;
  cusolverDnChegvd_bufferSize(
      handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
      jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int*/,
      a /*const cuComplex **/, lda /*int*/, b /*const cuComplex **/,
      ldb /*int*/, w /*const float **/, &buffer_size /*int **/);
  // End
}
