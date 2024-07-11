#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cudaStream_t s) {
  // Start
  cusolverDnSetStream(handle /*cusolverDnHandle_t*/, s /*cudaStream_t*/);
  // End
}
