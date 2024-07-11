#include <cudnn.h>

void test(cudnnBatchNormMode_t m, cudnnBatchNormOps_t op, void *diff_alphad,
          void *diff_betad, void *diff_alphap, void *diff_betap,
          cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnTensorDescriptor_t diff_summand_d, void *diff_summand,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src,
          cudnnTensorDescriptor_t p_d, void *scale, void *bias,
          void *diff_scale, void *diff_bias, double eps, void *smean,
          void *svar, cudnnActivationDescriptor_t adesc, void *workspace,
          size_t workspace_size, void *reservespace, size_t reservespace_size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnBatchNormalizationBackwardEx(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
      op /*cudnnBatchNormOps_t*/, diff_alphad /*void **/, diff_betad /*void **/,
      diff_alphap /*void **/, diff_betap /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      diff_summand_d /*cudnnTensorDescriptor_t*/, diff_summand /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
      p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      diff_scale /*void **/, diff_bias /*void **/, eps /*double*/,
      smean /*void **/, svar /*void **/, adesc /*cudnnActivationDescriptor_t*/,
      workspace /*void **/, workspace_size /*size_t*/, reservespace /*void **/,
      reservespace_size /*size_t*/);
  // End
}