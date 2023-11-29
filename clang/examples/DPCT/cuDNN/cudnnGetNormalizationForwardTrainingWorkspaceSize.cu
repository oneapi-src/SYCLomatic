#include <cudnn.h>

void test(cudnnNormMode_t m, cudnnNormOps_t op, cudnnNormAlgo_t alg,
          cudnnTensorDescriptor_t src_d, cudnnTensorDescriptor_t summand_d,
          cudnnTensorDescriptor_t dst_d, cudnnTensorDescriptor_t p1_d,
          cudnnActivationDescriptor_t adesc, cudnnTensorDescriptor_t p2_d,
          size_t *size, int group_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetNormalizationForwardTrainingWorkspaceSize(
      h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
      alg /*cudnnNormAlgo_t*/, src_d /*cudnnTensorDescriptor_t*/,
      summand_d /*cudnnTensorDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      p1_d /*cudnnTensorDescriptor_t*/, adesc /*cudnnActivationDescriptor_t*/,
      p2_d /*cudnnTensorDescriptor_t*/, size /*size_t **/, group_count /*int*/);
  // End
}