#include <cudnn.h>

void test(cudnnConvolutionDescriptor_t d) {
  // Start
  cudnnDestroyConvolutionDescriptor(d /*cudnnConvolutionDescriptor_t*/);
  // End
}