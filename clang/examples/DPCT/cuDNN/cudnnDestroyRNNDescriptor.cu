#include <cudnn.h>

void test(cudnnRNNDescriptor_t d) {
  // Start
  cudnnDestroyRNNDescriptor(d /*cudnnRNNDescriptor_t*/);
  // End
}