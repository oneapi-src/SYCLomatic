#include <cudnn.h>

void test(cudnnLRNDescriptor_t d) {
  // Start
  cudnnDestroyLRNDescriptor(d /*cudnnLRNDescriptor_t*/);
  // End
}