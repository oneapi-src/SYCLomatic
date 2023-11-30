#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnFilterDescriptor_t diff_filter_d,
          cudnnConvolutionBwdFilterAlgo_t alg, size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionBackwardFilterWorkspaceSize(
      h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_filter_d /*cudnnFilterDescriptor_t*/,
      alg /*cudnnConvolutionFwdAlgo_t*/, size /*size_t **/);
  // End
}