// Option: --use-experimental-features=level_zero

#include <cuda.h>

void test() {
  cudaIpcMemHandle_t *handle;
  void *ptr;
  // Start
  cudaIpcOpenMemHandle((void **)&ptr, *handle, cudaIpcMemLazyEnablePeerAccess);
  // End
}