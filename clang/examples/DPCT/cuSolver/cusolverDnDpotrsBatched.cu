#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t upper_lower, int n,
          int nrhs, double **a, int lda, double **b, int ldb, int *info,
          int group_count) {
  // Start
  cusolverDnDpotrsBatched(
      handle /*cusolverDnHandle_t*/, upper_lower /*cublasFillMode_t*/,
      n /*int*/, nrhs /*int*/, a /*double ***/, lda /*int*/, b /*double ***/,
      ldb /*int*/, info /*int **/, group_count /*int*/);
  // End
}
