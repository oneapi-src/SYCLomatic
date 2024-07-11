#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t filter_d,
          cudnnConvolutionDescriptor_t cdesc, cudnnTensorDescriptor_t dst_d,
          cudnnConvolutionFwdAlgo_t alg, size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetConvolutionForwardWorkspaceSize(
      h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
      filter_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      alg /*cudnnConvolutionFwdAlgo_t*/, size /*size_t **/);
  // End
}