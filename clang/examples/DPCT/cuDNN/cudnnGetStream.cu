#include <cudnn.h>

void test(cudaStream_t *s) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetStream(h /*cudnnHandle_t*/, s /*cudaStream_t **/);
  // End
}