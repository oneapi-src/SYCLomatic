#include <cudnn.h>

void test(size_t *size) {
  // Start
  cudnnTensorDescriptor_t d;
  cudnnGetTensorSizeInBytes(d /*cudnnTensorDescriptor_t*/, size /*size_t **/);
  // End
}