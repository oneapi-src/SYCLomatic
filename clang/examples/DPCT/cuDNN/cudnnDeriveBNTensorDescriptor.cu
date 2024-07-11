#include <cudnn.h>

void test(cudnnTensorDescriptor_t derived_desc, cudnnTensorDescriptor_t src_d,
          cudnnBatchNormMode_t m) {
  // Start
  cudnnDeriveBNTensorDescriptor(derived_desc /*cudnnTensorDescriptor_t*/,
                                src_d /*cudnnTensorDescriptor_t*/,
                                m /*cudnnBatchNormMode_t*/);
  // End
}