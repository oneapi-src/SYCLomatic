#include <cudnn.h>

void test(void *alpha, cudnnTensorDescriptor_t diff_dst_d, void *diff_dst,
          void *beta, cudnnTensorDescriptor_t diff_bias_d, void *diff_bias) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnConvolutionBackwardBias(h /*cudnnHandle_t*/, alpha /*void **/,
                               diff_dst_d /*cudnnTensorDescriptor_t*/,
                               diff_dst /*void **/, beta /*void **/,
                               diff_bias_d /*cudnnTensorDescriptor_t*/,
                               diff_bias /*void **/);
  // End
}