#include <cudnn.h>

void test(size_t *size) {
  // Start
  cudnnFilterDescriptor_t d;
  cudnnGetFilterSizeInBytes(d /*cudnnFilterDescriptor_t*/, size /*size_t **/);
  // End
}