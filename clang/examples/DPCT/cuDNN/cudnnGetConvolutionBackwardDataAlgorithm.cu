#include <cudnn.h>

void test(cudnnFilterDescriptor_t filter_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnTensorDescriptor_t diff_src_d,
          cudnnConvolutionBwdDataPreference_t preference, size_t limit,
          cudnnConvolutionBwdDataAlgo_t *alg) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionBackwardDataAlgorithm(
      h /*cudnnHandle_t*/, filter_d /*cudnnFilterDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_src_d /*cudnnTensorDescriptor_t*/,
      preference /*cudnnConvolutionBwdDataPreference_t*/, limit /*size_t*/,
      alg /*cudnnConvolutionBwdDataAlgo_t **/);
  // End
}