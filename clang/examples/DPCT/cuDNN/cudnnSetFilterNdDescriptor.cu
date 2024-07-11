#include <cudnn.h>

void test(cudnnDataType_t t, cudnnTensorFormat_t f, int n, int da[]) {
  // Start
  cudnnFilterDescriptor_t d;
  cudnnSetFilterNdDescriptor(d /*cudnnFilterDescriptor_t*/,
                             t /*cudnnDataType_t*/, f /*cudnnTensorFormat_t*/,
                             n /*int*/, da /*int[]*/);
  // End
}