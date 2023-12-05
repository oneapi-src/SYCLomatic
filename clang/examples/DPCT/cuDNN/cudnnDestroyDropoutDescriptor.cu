#include <cudnn.h>

void test(cudnnDropoutDescriptor_t d) {
  // Start
  cudnnDestroyDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/);
  // End
}