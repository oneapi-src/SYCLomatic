#include <cudnn.h>

void test(cudnnFilterDescriptor_t filter_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnTensorDescriptor_t diff_src_d, cudnnConvolutionBwdDataAlgo_t alg,
          size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionBackwardDataWorkspaceSize(
      h /*cudnnHandle_t*/, filter_d /*cudnnTensorDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_src_d /*cudnnTensorDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
      size /*size_t **/);
  // End
}