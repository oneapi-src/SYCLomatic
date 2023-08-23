// Migration desc: The API is Removed.
#include <cudnn.h>

void test(cudnnTensorDescriptor_t *d) {
  // Start
  cudnnCreateTensorDescriptor(d /*cudnnTensorDescriptor_t **/);
  // End
}
