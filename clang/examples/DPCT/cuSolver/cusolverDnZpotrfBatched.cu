#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t upper_lower, int n,
          cuDoubleComplex **a, int lda, int *info, int group_count) {
  // Start
  cusolverDnZpotrfBatched(handle /*cusolverDnHandle_t*/,
                          upper_lower /*cublasFillMode_t*/, n /*int*/,
                          a /*cuDoubleComplex ***/, lda /*int*/, info /*int **/,
                          group_count /*int*/);
  // End
}
