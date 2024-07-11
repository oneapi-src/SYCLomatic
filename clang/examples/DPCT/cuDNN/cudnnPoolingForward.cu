#include <cudnn.h>

void test(cudnnPoolingDescriptor_t desc, void *alpha,
          cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnPoolingForward(h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/,
                      alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
                      src /*void **/, beta /*void **/,
                      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}