#include <cudnn.h>

void test(cudnnRNNDescriptor_t d, size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetRNNWeightSpaceSize(h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/,
                             size /*size_t **/);
  // End
}