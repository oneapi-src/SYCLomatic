#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t upper_lower, int n,
          int nrhs, float **a, int lda, float **b, int ldb, int *info,
          int group_count) {
  // Start
  cusolverDnSpotrsBatched(
      handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
      n /*int*/, nrhs /*int*/, a /*float ***/, lda /*int*/, b /*float ***/,
      ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
