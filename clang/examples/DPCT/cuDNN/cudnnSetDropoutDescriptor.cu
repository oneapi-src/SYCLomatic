#include <cudnn.h>

void test(cudnnHandle_t h, float dropout, void *states, size_t statesize,
          unsigned long long seed) {
  // Start
  cudnnDropoutDescriptor_t d;
  cudnnSetDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/, h /*cudnnHandle_t*/,
                            dropout /*float*/, states /*void **/,
                            statesize /*size_t*/, seed /*unsigned long long*/);
  // End
}