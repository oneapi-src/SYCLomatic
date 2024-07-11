#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, int *n, int *c, int *h, int *w) {
  // Start
  cudnnPoolingDescriptor_t d;
  cudnnGetPooling2dForwardOutputDim(
      d /*cudnnPoolingDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
      n /*int**/, c /*int**/, h /*int**/, w /*int**/);
  // End
}