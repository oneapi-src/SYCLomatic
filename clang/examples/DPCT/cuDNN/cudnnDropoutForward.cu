#include <cudnn.h>

void test(cudnnDropoutDescriptor_t d, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t dst_d, void *dst, void *reservespace,
          size_t reservespace_size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnDropoutForward(h /*cudnnHandle_t*/, d /*cudnnDropoutDescriptor_t*/,
                      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
                      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
                      reservespace /*void **/, reservespace_size /*size_t*/);
  // End
}