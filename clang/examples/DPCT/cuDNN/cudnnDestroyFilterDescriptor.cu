#include <cudnn.h>

void test(cudnnFilterDescriptor_t d) {
  // Start
  cudnnDestroyFilterDescriptor(d /*cudnnFilterDescriptor_t*/);
  // End
}