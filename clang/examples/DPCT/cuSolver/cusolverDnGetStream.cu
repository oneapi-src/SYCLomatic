#include "cusolverDn.h"

void test(cusolverDnHandle_t handle) {
  // Start
  cudaStream_t s;
  cusolverDnGetStream(handle /*cusolverDnHandle_t*/, &s /*cudaStream_t **/);
  // End
}
