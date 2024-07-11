#include <cudnn.h>

void test(cudnnBatchNormMode_t m, void *alpha, void *beta,
          cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t dst_d, void *dst, cudnnTensorDescriptor_t p_d,
          void *scale, void *bias, void *mean, void *var, double eps) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnBatchNormalizationForwardInference(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/, alpha /*void **/,
      beta /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
      p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      mean /*void **/, var /*void **/, eps /*double*/);
  // End
}