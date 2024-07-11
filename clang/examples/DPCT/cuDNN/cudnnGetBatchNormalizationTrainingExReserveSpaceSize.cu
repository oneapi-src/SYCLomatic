#include <cudnn.h>

void test(cudnnBatchNormMode_t m, cudnnBatchNormOps_t op,
          cudnnActivationDescriptor_t adesc, cudnnTensorDescriptor_t src_d,
          size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
      op /*cudnnBatchNormOps_t*/, adesc /*cudnnActivationDescriptor_t*/,
      src_d /*cudnnTensorDescriptor_t*/, size /*size_t **/);
  // End
}