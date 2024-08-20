#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
          const double *a, int lda, const double *tau) {
  // Start
  int buffer_size;
  cusolverDnDorgtr_bufferSize(handle /*cusolverDnHandle_t*/,
                              uplo /*cublasFillMode_t*/, n /*int*/,
                              a /*const double **/, lda /*int*/,
                              tau /*const double **/, &buffer_size /*int **/);
  // End
}
