#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t filter_d,
          cudnnConvolutionDescriptor_t cdesc, cudnnTensorDescriptor_t dst_d,
          cudnnConvolutionFwdPreference_t preference, size_t limit,
          cudnnConvolutionFwdAlgo_t *alg) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionForwardAlgorithm(
      h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
      filter_d /*cudnnFilterDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      preference /*cudnnConvolutionFwdPreference_t*/, limit /*size_t*/,
      alg /*cudnnConvolutionFwdAlgo_t **/);
  // End
}