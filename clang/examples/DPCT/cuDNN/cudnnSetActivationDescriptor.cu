#include <cudnn.h>

void test(cudnnActivationMode_t m, cudnnNanPropagation_t p, double c) {
  // Start
  cudnnActivationDescriptor_t d;
  cudnnSetActivationDescriptor(d /*cudnnActivationDescriptor_t*/,
                               m /*cudnnActivationMode_t*/,
                               p /*cudnnNanPropagation_t*/, c /*double*/);
  // End
}
