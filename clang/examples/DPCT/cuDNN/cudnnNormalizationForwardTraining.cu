#include <cudnn.h>

void test(cudnnNormMode_t m, cudnnNormOps_t op, cudnnNormAlgo_t alg,
          void *alpha, void *beta, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t p1_d, void *scale, void *bias, double factor,
          cudnnTensorDescriptor_t p2_d, void *rmean, void *rvar, double eps,
          void *smean, void *svar, cudnnActivationDescriptor_t adesc,
          cudnnTensorDescriptor_t summand_d, void *summand,
          cudnnTensorDescriptor_t dst_d, void *dst, void *workspace,
          size_t workspace_size, void *reservespace,
          size_t reservespace_size, int group_count) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnNormalizationForwardTraining(
      h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
      alg /*cudnnNormAlgo_t*/, alpha /*void **/, beta /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      factor /*double*/, p2_d /*cudnnTensorDescriptor_t*/, rmean /*void **/,
      rvar /*void **/, eps /*double*/, smean /*void **/, svar /*void **/,
      adesc /*cudnnActivationDescriptor_t*/,
      summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/, workspace /*void **/,
      workspace_size /*size_t*/, reservespace /*void **/,
      reservespace_size /*size_t*/, group_count /*int*/);
  // End
}