#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverEigType_t itype,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex *a,
          int lda, cuComplex *b, int ldb, float *w, cuComplex *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnChegvd(handle /*cusolverDnHandle_t*/, itype /*cusolverEigType_t*/,
                   jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuComplex **/, lda /*int*/, b /*cuComplex **/,
                   ldb /*int*/, w /*float **/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
