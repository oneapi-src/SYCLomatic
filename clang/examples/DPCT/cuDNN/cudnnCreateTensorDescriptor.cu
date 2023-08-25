// Migration desc: The API is Removed.
#include <cudnn.h>

void test() {
  // Start
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d /*cudnnTensorDescriptor_t **/);
  // End
}
