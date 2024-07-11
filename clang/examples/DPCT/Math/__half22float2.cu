// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(__half2 h) {
  // Start
  __half22float2(h /*__half2*/);
  // End
}
