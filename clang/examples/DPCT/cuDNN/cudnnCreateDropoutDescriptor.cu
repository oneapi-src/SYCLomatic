#include <cudnn.h>

void test() {
  // Start
  cudnnDropoutDescriptor_t d;
  cudnnCreateDropoutDescriptor(&d /*cudnnDropoutDescriptor_t **/);
  // End
}