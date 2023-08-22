#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t upper_lower, int n,
          int nrhs, cuDoubleComplex **a, int lda, cuDoubleComplex **b, int ldb,
          int *info, int group_count) {
  // Start
  cusolverDnZpotrsBatched(handle /*cusolverDnHandle_t*/,
                          upper_lower /*cublasFillMode_t*/, n /*int*/,
                          nrhs /*int*/, a /*cuDoubleComplex ***/, lda /*int*/,
                          b /*cuDoubleComplex ***/, ldb /*int*/, info /*int **/,
                          group_count /*int*/);
  // End
}
