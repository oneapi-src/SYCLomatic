#include <cudnn.h>

void test(cudnnFilterDescriptor_t weight_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnTensorDescriptor_t diff_src_d, int reqc, int *realc) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBwdDataAlgoPerf_t r;
  cudnnGetConvolutionBackwardDataAlgorithm_v7(
      h /*cudnnHandle_t*/, weight_d /*cudnnFilterDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_src_d /*cudnnTensorDescriptor_t*/, reqc /*int*/, realc /*int **/,
      &r /*cudnnConvolutionBwdDataAlgoPerf_t*/);
  // End
}