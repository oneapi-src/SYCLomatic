#include <cudnn.h>

void test(cudnnNormMode_t m, cudnnNormOps_t op, cudnnNormAlgo_t alg,
          void *alpha, void *beta, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t p1_d, void *scale, void *bias,
          cudnnTensorDescriptor_t p2_d, void *emean, void *evar,
          cudnnTensorDescriptor_t summand_d, void *summand,
          cudnnActivationDescriptor_t adesc, cudnnTensorDescriptor_t dst_d,
          void *dst, double eps, int group_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnNormalizationForwardInference(
      h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
      alg /*cudnnNormAlgo_t*/, alpha /*void **/, beta /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      p2_d /*cudnnTensorDescriptor_t*/, emean /*void **/, evar /*void **/,
      summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
      adesc /*cudnnActivationDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
      dst /*void **/, eps /*double*/, group_count /*int*/);
  // End
}