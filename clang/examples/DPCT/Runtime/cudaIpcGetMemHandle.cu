// Option: --use-experimental-features=level_zero
#include <cuda.h>

void test() {
  cudaIpcMemHandle_t *handle;
  void *ptr;
  // Start
  cudaIpcGetMemHandle(handle/*cudaIpcMemHandle_t **/, ptr/*void **/);
  // End
}