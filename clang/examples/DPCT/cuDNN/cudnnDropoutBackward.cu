#include <cudnn.h>

void test(cudnnDropoutDescriptor_t d, cudnnTensorDescriptor_t diff_dst_d,
          void *diff_dst, cudnnTensorDescriptor_t diff_src_d, void *diff_src,
          void *reservespace, size_t reservespace_size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnDropoutBackward(
      h /*cudnnHandle_t*/, d /*cudnnDropoutDescriptor_t*/,
      diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
      diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
      reservespace /*void **/, reservespace_size /*size_t*/);
  // End
}