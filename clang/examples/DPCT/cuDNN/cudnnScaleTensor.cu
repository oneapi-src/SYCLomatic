#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, void *src, void *factor) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnScaleTensor(h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
                   src /*void **/, factor /*void **/);
  // End
}