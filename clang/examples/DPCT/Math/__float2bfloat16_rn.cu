// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(float f) {
  // Start
  __float2bfloat16_rn(f /*float*/);
  // End
}
