// Option: --use-experimental-features=level_zero
#include <cuda.h>

void test() {
  void *ptr;
  // Start
  cudaIpcCloseMemHandle(ptr/*void **/);
  // End
}