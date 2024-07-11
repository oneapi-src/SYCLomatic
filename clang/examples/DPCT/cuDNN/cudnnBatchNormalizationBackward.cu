#include <cudnn.h>

void test(cudnnBatchNormMode_t m, void *alphad, void *betad, void *alphap,
          void *betap, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src,
          cudnnTensorDescriptor_t p_d, void *scale, void *diff_scale,
          void *diff_bias, double eps, void *smean, void *svar) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnBatchNormalizationBackward(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/, alphad /*void **/,
      betad /*void **/, alphap /*void **/, betap /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
      p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, diff_scale /*void **/,
      diff_bias /*void **/, eps /*double*/, smean /*void **/, svar /*void **/);
  // End
}