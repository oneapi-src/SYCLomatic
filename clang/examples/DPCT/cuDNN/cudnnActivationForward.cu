#include <cudnn.h>

void test(cudnnActivationDescriptor_t desc, void *alpha, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src,
          cudnnTensorDescriptor_t src_d, void *src) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnActivationForward(
      h /*cudnnHandle_t*/, desc /*cudnnActivationDescriptor_t*/,
      alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}
