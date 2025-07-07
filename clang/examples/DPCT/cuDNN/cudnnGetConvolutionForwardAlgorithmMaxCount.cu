#include <cudnn.h>

void test(int max_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionForwardAlgorithmMaxCount(h /*cudnnHandle_t */, &max_count /*int **/);
  // End
}