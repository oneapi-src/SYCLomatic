#include <cudnn.h>

void test(int *group_count) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnGetConvolutionGroupCount(d /*cudnnActivationDescriptor_t*/,
                                group_count /*int**/);
  // End
}