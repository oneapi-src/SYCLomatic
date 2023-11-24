#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, int n, int da[]) {
  // Start
  cudnnPoolingDescriptor_t d;
  cudnnGetPoolingNdForwardOutputDim(d /*cudnnPoolingDescriptor_t*/,
                                    src_d /*cudnnTensorDescriptor_t*/,
                                    n /*int*/, da /*int[]*/);
  // End
}