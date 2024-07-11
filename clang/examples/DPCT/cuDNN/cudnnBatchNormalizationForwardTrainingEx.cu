#include <cudnn.h>

void test(cudnnBatchNormMode_t m, cudnnBatchNormOps_t op, void *alpha,
          void *beta, cudnnTensorDescriptor_t src_d, void *src,
          cudnnTensorDescriptor_t summand_d, void *summand,
          cudnnTensorDescriptor_t dst_d, void *dst, cudnnTensorDescriptor_t p_d,
          void *scale, void *bias, double factor, void *rmean, void *rvar,
          double eps, void *smean, void *svar,
          cudnnActivationDescriptor_t adesc, void *workspace,
          size_t workspace_size, void *reservespace, size_t reservespace_size) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnBatchNormalizationForwardTrainingEx(
      h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
      op /*cudnnBatchNormOps_t*/, alpha /*void **/, beta /*void **/,
      src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
      summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
      p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
      factor /*double*/, rmean /*void **/, rvar /*void **/, eps /*double*/,
      smean /*void **/, svar /*void **/, adesc /*cudnnActivationDescriptor_t*/,
      workspace /*void **/, workspace_size /*size_t*/, reservespace /*void **/,
      reservespace_size /*size_t*/);
  // End
}