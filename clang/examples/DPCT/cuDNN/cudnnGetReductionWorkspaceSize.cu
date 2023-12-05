#include <cudnn.h>

void test(cudnnReduceTensorDescriptor_t d, cudnnTensorDescriptor_t src_d, 
    cudnnTensorDescriptor_t dst_d, size_t *size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetReductionWorkspaceSize(
      h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
      src_d /*cudnnTensorDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      size /*size_t **/);
  // End
}