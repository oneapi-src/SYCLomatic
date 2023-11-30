#include <cudnn.h>

void test(cudnnActivationDescriptor_t d) {
  // Start
  cudnnDestroyActivationDescriptor(d /*cudnnActivationDescriptor_t*/);
  // End
}