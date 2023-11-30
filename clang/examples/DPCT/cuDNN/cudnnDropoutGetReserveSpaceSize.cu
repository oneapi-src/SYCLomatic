#include <cudnn.h>

void test(cudnnTensorDescriptor_t src_d, size_t *size) {
  // Start
  cudnnDropoutGetReserveSpaceSize(src_d /*cudnnTensorDescriptor_t*/,
                                  size /*size_t **/);
  // End
}