#include <cudnn.h>

void test(cudnnActivationDescriptor_t d, cudnnActivationMode_t m,
          cudnnNanPropagation_t p, double c) {
  // Start
  cudnnSetActivationDescriptor(d /*cudnnActivationDescriptor_t*/,
                               m /*cudnnActivationMode_t*/,
                               p /*cudnnNanPropagation_t*/, c /*double*/);
  // End
}
