#include <cudnn.h>

void test(cudnnRNNDataDescriptor_t d) {
  // Start
  cudnnDestroyRNNDataDescriptor(d /*cudnnRNNDataDescriptor_t*/);
  // End
}