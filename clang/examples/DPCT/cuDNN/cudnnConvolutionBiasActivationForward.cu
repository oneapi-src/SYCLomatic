#include <cudnn.h>

void test(void *alpha1, cudnnTensorDescriptor_t src_d, void *src,
          cudnnFilterDescriptor_t filter_d, void *filter,
          cudnnConvolutionDescriptor_t cdesc, cudnnConvolutionFwdAlgo_t alg,
          void *workspace, size_t workspace_size, void *alpha2,
          cudnnTensorDescriptor_t summand_d, void *summand,
          cudnnTensorDescriptor_t bias_d, void *bias,
          cudnnActivationDescriptor_t adesc, cudnnTensorDescriptor_t dst_d,
          void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBiasActivationForward(
      h /*cudnnHandle_t*/, alpha1 /*void **/, src_d /*cudnnTensorDescriptor_t*/,
      src /*void **/, filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
      cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
      workspace /*void **/, workspace_size /*size_t*/, alpha2 /*void **/,
      summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
      bias_d /*cudnnTensorDescriptor_t*/, bias /*void **/,
      adesc /*cudnnActivationDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      dst /*void **/);
  // End
}