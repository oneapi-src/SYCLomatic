#include <cudnn.h>

void test(cudnnTensorDescriptor_t d, cudnnTensorFormat_t f, cudnnDataType_t t,
          int n, int c, int h, int w) {
  // Start
  cudnnSetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
                             f /*cudnnTensorFormat_t*/, t /*cudnnDataType_t*/,
                             n /*int*/, c /*int*/, h /*int*/, w /*int*/);
  // End
}
