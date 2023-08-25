// Migration desc: The API is Removed.
#include <cudnn.h>

void test() {
  // Start
  cudnnActivationDescriptor_t d;
  cudnnCreateActivationDescriptor(&d /*cudnnActivationDescriptor_t **/);
  // End
}
