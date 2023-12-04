#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnFilterDescriptor_t filter_d,
          cudnnConvolutionDescriptor_t cdesc, cudnnTensorDescriptor_t dst_d,
          int reqc, int *realc) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionFwdAlgoPerf_t r;
  cudnnFindConvolutionForwardAlgorithm(
      h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
      filter_d /*cudnnFilterDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      reqc /*int*/, realc /*int **/, &r /*cudnnConvolutionFwdAlgoPerf_t*/);
  // End
}