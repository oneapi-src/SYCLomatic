#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t upper_lower, int n,
          int nrhs, cuComplex **a, int lda, cuComplex **b, int ldb, int *info,
          int group_count) {
  // Start
  cusolverDnCpotrsBatched(
      handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
      n /*int*/, nrhs /*int*/, a /*cuComplex ***/, lda /*int*/,
      b /*cuComplex ***/, ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
