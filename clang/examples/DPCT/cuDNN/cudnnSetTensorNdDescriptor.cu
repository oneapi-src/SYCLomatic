#include <cudnn.h>

void test(cudnnDataType_t t, int nd, int da[], int sa[]) {
  // Start
  cudnnTensorDescriptor_t d;
  cudnnSetTensorNdDescriptor(d /*cudnnTensorDescriptor_t*/,
                             t /*cudnnDataType_t*/, nd /*int*/, da /*int[]*/,
                             sa /*int[]*/);
  // End
}