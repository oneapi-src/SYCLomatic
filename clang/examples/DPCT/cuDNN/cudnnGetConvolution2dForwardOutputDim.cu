#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t f_d, int *n,
          int *c, int *h, int *w) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnGetConvolution2dForwardOutputDim(
      d /*cudnnConvolutionDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
      f_d /*cudnnFilterDescriptor_t*/, n /*int**/, c /*int**/, h /*int**/,
      w /*int**/);
  // End
}