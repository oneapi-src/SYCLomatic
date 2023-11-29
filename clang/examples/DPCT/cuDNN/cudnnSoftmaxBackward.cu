#include <cudnn.h>

void test(cudnnSoftmaxAlgorithm_t a, cudnnSoftmaxMode_t m, void *alpha,
          cudnnTensorDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t diff_dst_d, void *diff_dst, void *beta,
          cudnnTensorDescriptor_t diff_src_d, void *diff_src) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnSoftmaxBackward(h /*cudnnHandle_t*/, a /*cudnnSoftmaxAlgorithm_t*/,
                       m /*cudnnSoftmaxMode_t*/, alpha /*void **/,
                       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
                       diff_dst_d /*cudnnTensorDescriptor_t*/,
                       diff_dst /*void **/, beta /*void **/,
                       diff_src_d /*cudnnTensorDescriptor_t*/,
                       diff_src /*void **/);
  // End
}