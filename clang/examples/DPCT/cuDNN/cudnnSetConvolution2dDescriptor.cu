#include <cudnn.h>

void test(int padding_h, int padding_w, int stride_h, int stride_w,
          int dilation_h, int dilation_w, cudnnConvolutionMode_t m,
          cudnnDataType_t t) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnSetConvolution2dDescriptor(
      d /*cudnnConvolutionDescriptor_t*/, padding_h /*int*/, padding_w /*int*/,
      stride_h /*int*/, stride_w /*int*/, dilation_h /*int*/,
      dilation_w /*int*/, m /*cudnnConvolutionMode_t*/, t /*cudnnDataType_t*/);
  // End
}