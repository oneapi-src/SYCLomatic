#include <cudnn.h>

void test(int max_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionBackwardDataAlgorithmMaxCount(h /*cudnnHandle_t */, &max_count /*int **/);
  // End
}
