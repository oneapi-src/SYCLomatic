#include <cudnn.h>

void test(cudnnHandle_t h, float *dropout, void **states,
          unsigned long long *seed) {
  // Start
  cudnnDropoutDescriptor_t d;
  cudnnGetDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/, h /*cudnnHandle_t*/,
                            dropout /*float **/, states /*void ***/,
                            seed /*unsigned long long **/);
  // End
}