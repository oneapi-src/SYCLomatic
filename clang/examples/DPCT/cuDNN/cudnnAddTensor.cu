#include <cudnn.h>

void test(void *alpha, cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnAddTensor(h /*cudnnHandle_t*/, alpha /*void **/,
                 src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
                 beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/,
                 dst /*void **/);
  // End
}