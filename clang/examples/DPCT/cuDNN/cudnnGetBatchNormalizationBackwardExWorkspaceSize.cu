#include <cudnn.h>

void test(cudnnBatchNormMode_t m, cudnnBatchNormOps_t op,
          cudnnTensorDescriptor_t src_d, cudnnTensorDescriptor_t dst_d,
          cudnnTensorDescriptor_t diff_dst_d,
          cudnnTensorDescriptor_t diff_summand_d,
          cudnnTensorDescriptor_t diff_src_d, cudnnTensorDescriptor_t p_d,
          cudnnActivationDescriptor_t adesc, size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
      op /*cudnnBatchNormOps_t*/, src_d /*cudnnTensorDescriptor_t*/,
      dst_d /*cudnnTensorDescriptor_t*/, diff_dst_d /*cudnnTensorDescriptor_t*/,
      diff_summand_d /*cudnnTensorDescriptor_t*/,
      diff_src_d /*cudnnTensorDescriptor_t*/, p_d /*cudnnTensorDescriptor_t*/,
      adesc /*cudnnActivationDescriptor_t*/, size /*size_t **/);
  // End
}