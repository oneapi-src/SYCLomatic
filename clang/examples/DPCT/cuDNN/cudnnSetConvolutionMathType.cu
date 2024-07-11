#include <cudnn.h>

void test(cudnnMathType_t mt) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnSetConvolutionMathType(d /*cudnnActivationDescriptor_t*/,
                              mt /*cudnnMathType_t*/);
  // End
}