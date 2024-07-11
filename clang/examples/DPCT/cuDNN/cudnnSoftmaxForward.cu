#include <cudnn.h>

void test(cudnnSoftmaxAlgorithm_t a, cudnnSoftmaxMode_t m, void *alpha,
          cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnSoftmaxForward(h /*cudnnHandle_t*/, a /*cudnnSoftmaxAlgorithm_t*/,
                      m /*cudnnSoftmaxMode_t*/, alpha /*void **/,
                      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
                      beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/,
                      dst /*void **/);
  // End
}