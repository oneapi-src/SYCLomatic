#include <cudnn.h>

void test(cudnnHandle_t h) {
  // Start
  cudnnDestroy(h /*cudnnHandle_t*/);
  // End
}
