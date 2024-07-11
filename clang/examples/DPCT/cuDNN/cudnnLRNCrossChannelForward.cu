#include <cudnn.h>

void test(cudnnLRNDescriptor_t desc, cudnnLRNMode_t m, void *alpha,
          cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnLRNCrossChannelForward(
      h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/, m /*cudnnLRNMode_t*/,
      alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}