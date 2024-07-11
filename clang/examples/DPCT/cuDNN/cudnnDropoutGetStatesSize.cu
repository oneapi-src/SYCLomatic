#include <cudnn.h>

void test(size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnDropoutGetStatesSize(h /*cudnnHandle_t*/, size /*size_t **/);
  // End
}