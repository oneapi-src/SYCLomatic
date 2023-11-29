#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t f_d, int n,
          int da[]) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnGetConvolutionNdForwardOutputDim(
      d /*cudnnPoolingDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
      f_d /*cudnnTensorDescriptor_t*/, n /*int*/, da /*int[]*/);
  // End
}