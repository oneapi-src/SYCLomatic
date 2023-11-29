#include <cudnn.h>

void test(cudnnDataType_t t, cudnnRNNDataLayout_t l, int len, int b, int v,
          int sa[], void *p) {
  // Start
  cudnnRNNDataDescriptor_t d;
  cudnnSetRNNDataDescriptor(d /*cudnnTensorDescriptor_t*/,
                            t /*cudnnDataType_t*/, l /*cudnnRNNDataLayout_t*/,
                            len /*int*/, b /*int*/, v /*int*/, sa /*int[]*/,
                            p /*void **/);
  // End
}