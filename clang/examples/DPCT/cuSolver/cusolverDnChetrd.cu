#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *a,
          int lda, float *d, float *e, cuComplex *tau, cuComplex *buffer,
          int buffer_size, int *info) {
  // Start
  cusolverDnChetrd(handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
                   n /*int*/, a /*cuComplex **/, lda /*int*/, d /*float **/,
                   e /*float **/, tau /*cuComplex **/, buffer /*cuComplex **/,
                   buffer_size /*int*/, info /*int **/);
  // End
}
