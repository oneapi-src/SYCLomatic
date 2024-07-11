#include <cudnn.h>

void test(void *alpha, cudnnTensorDescriptor_t src_d, void *src,
          cudnnFilterDescriptor_t filter_d, void *filter,
          cudnnConvolutionDescriptor_t cdesc, cudnnConvolutionFwdAlgo_t alg,
          void *workspace, size_t workspace_size, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionForward(
      h /*cudnnHandle_t*/, alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
      src /*void **/, filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
      cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
      workspace /*void **/, workspace_size /*size_t*/, beta /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}