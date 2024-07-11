#include <cudnn.h>

void test(cudnnLRNDescriptor_t desc, cudnnLRNMode_t m, void *alpha,
          cudnnTensorDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnLRNCrossChannelBackward(
      h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/, m /*cudnnLRNMode_t*/,
      alpha /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/, beta /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
  // End
}