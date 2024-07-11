#include <cudnn.h>

void test(cudnnOpTensorDescriptor_t d) {
  // Start
  cudnnDestroyOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t*/);
  // End
}