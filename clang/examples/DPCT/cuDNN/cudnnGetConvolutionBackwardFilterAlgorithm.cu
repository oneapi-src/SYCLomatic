#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnFilterDescriptor_t diff_filter_d,
          cudnnConvolutionBwdFilterPreference_t preference, size_t limit,
          cudnnConvolutionBwdFilterAlgo_t *alg) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionBackwardFilterAlgorithm(
      h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_filter_d /*cudnnFilterDescriptor_t*/,
      preference /*cudnnConvolutionBwdFilterPreference_t*/, limit /*size_t*/,
      alg /*cudnnConvolutionBwdFilterAlgo_t **/);
  // End
}