// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(float f) {
  // Start
  __float2half_rz(f /*float*/);
  // End
}
