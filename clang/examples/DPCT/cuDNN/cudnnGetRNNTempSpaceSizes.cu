#include <cudnn.h>

void test(cudnnHandle_t handle, cudnnRNNDescriptor_t d, cudnnForwardMode_t m,
          cudnnRNNDataDescriptor_t src_d, size_t *workspace_size,
          size_t *reservespace_size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnGetRNNTempSpaceSizes(
      h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
      m /*cudnnReduceTensorDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
      workspace_size /*size_t **/, reservespace_size /*size_t **/);
  // End
}