#include <cudnn.h>

void test(cudnnTensorDescriptor_t d) {
  // Start
  cudnnDestroyTensorDescriptor(d /*cudnnTensorDescriptor_t*/);
  // End
}