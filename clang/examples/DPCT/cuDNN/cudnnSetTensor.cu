#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, void *src, void *value) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnSetTensor(h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
                 src /*void **/, value /*void **/);
  // End
}