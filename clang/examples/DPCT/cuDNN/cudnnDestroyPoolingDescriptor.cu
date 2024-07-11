#include <cudnn.h>

void test(cudnnPoolingDescriptor_t d) {
  // Start
  cudnnDestroyPoolingDescriptor(d /*cudnnPoolingDescriptor_t*/);
  // End
}