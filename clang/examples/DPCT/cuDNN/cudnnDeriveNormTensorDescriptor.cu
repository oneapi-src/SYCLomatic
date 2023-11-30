#include <cudnn.h>

void test(cudnnTensorDescriptor_t derived_p1_desc,
          cudnnTensorDescriptor_t derived_p2_desc,
          cudnnTensorDescriptor_t src_d, cudnnNormMode_t m, int group_count) {
  // Start
  cudnnDeriveNormTensorDescriptor(derived_p1_desc /*cudnnTensorDescriptor_t*/,
                                  derived_p2_desc /*cudnnTensorDescriptor_t*/,
                                  src_d /*cudnnTensorDescriptor_t*/,
                                  m /*cudnnNormMode_t*/, group_count /*int*/);
  // End
}