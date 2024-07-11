#include <cudnn.h>

void test(cudnnReduceTensorDescriptor_t d) {
  // Start
  cudnnDestroyReduceTensorDescriptor(d /*cudnnReduceTensorDescriptor_t*/);
  // End
}