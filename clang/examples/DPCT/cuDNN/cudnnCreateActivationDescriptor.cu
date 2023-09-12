// Migration desc: The API is Removed.
#include <cudnn.h>

void test(cudnnActivationDescriptor_t *d) {
  // Start
  cudnnCreateActivationDescriptor(d /*cudnnActivationDescriptor_t **/);
  // End
}
