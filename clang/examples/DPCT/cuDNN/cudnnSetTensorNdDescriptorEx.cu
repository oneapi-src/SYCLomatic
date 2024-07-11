#include <cudnn.h>

void test(cudnnTensorFormat_t f, cudnnDataType_t t, int nd, int da[]) {
  // Start
  cudnnTensorDescriptor_t d;
  cudnnSetTensorNdDescriptorEx(d /*cudnnTensorDescriptor_t*/,
                               f /*cudnnTensorFormat_t*/, t /*cudnnDataType_t*/,
                               nd /*int*/, da /*int[]*/);
  // End
}