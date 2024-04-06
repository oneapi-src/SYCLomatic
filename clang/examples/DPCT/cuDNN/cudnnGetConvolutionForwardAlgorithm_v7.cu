#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t weight_d,
          cudnnConvolutionDescriptor_t cdesc, cudnnTensorDescriptor_t dst_d,
          int reqc, int *realc) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionFwdAlgoPerf_t r;
  cudnnGetConvolutionForwardAlgorithm_v7(
      h /*cudnnHandle_t*/, src_d /*cudnnFilterDescriptor_t*/,
      weight_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnFilterDescriptor_t*/,
      reqc /*int*/, realc /*int **/, &r /*cudnnConvolutionFwdAlgoPerf_t*/);
  // End
}