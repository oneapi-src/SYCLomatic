#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, cudnnTensorDescriptor_t diff_dst_d,
          cudnnConvolutionDescriptor_t cdesc,
          cudnnFilterDescriptor_t diff_weight_d, int reqc, int *realc) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBwdFilterAlgoPerf_t r;
  cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      h /*cudnnHandle_t*/, src_d /*cudnnFilterDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/,
      cdesc /*cudnnConvolutionDescriptor_t*/,
      diff_weight_d /*cudnnFilterDescriptor_t*/, reqc /*int*/, realc /*int **/,
      &r /*cudnnConvolutionBwdFilterAlgoPerf_t*/);
  // End
}