#include <cudnn.h>

void test(cudnnNormMode_t m, cudnnNormOps_t op, cudnnNormAlgo_t alg,
          void *diff_alphad, void *diff_betad, void *diff_alphap,
          void *diff_betap, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnTensorDescriptor_t diff_summand_d, void *diff_summand,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src,
          cudnnTensorDescriptor_t p1_d, void *scale, void *bias,
          void *diff_scale, void *diff_bias, double eps,
          cudnnTensorDescriptor_t p2_d, void *smean, void *svar,
          cudnnActivationDescriptor_t adesc, void *workspace,
          size_t workspace_size, void *reservespace, size_t reservespace_size,
          int group_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnNormalizationBackward(
      h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
      alg /*cudnnNormAlgo_t*/, diff_alphad /*void **/, diff_betad /*void **/,
      diff_alphap /*void **/, diff_betap /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      diff_summand_d /*cudnnTensorDescriptor_t*/, diff_summand /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
      p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      diff_scale /*void **/, diff_bias /*void **/, eps /*double*/,
      p2_d /*cudnnTensorDescriptor_t*/, smean /*void **/, svar /*void **/,
      adesc /*cudnnActivationDescriptor_t*/, workspace /*void **/,
      workspace_size /*size_t*/, reservespace /*void **/,
      reservespace_size /*size_t*/, group_count /*int*/);
  // End
}