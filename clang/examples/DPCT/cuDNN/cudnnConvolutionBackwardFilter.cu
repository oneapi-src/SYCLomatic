#include <cudnn.h>

void test(void *alpha, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnConvolutionBwdFilterAlgo_t alg, void *workspace,
          size_t workspace_size, void *beta,
          cudnnFilterDescriptor_t diff_filter_d, void *diff_filter) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBackwardFilter(
      h /*cudnnHandle_t*/, alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
      src /*void **/, diff_dst_d /*cudnnTensorDescriptor_t*/,
      diff_dst /*void **/, cdesc /*cudnnConvolutionDescriptor_t*/,
      alg /*cudnnConvolutionFwdAlgo_t*/, workspace /*void **/,
      workspace_size /*size_t*/, beta /*void **/,
      diff_filter_d /*cudnnTensorDescriptor_t*/, diff_filter /*void **/);
  // End
}