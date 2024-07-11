#include <cudnn.h>

void test(void *alpha, cudnnFilterDescriptor_t filter_d, void *filter,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnConvolutionDescriptor_t cdesc, cudnnConvolutionBwdDataAlgo_t alg,
          void *workspace, size_t workspace_size, void *beta,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBackwardData(
      h /*cudnnHandle_t*/, alpha /*void **/,
      filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
      workspace /*void **/, workspace_size /*size_t*/, beta /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
  // End
}