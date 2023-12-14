// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(float2 f) {
  // Start
  __float22half2_rn(f /*float2*/);
  // End
}
