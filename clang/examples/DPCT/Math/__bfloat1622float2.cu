// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(__nv_bfloat162 b) {
  // Start
  __bfloat1622float2(b /*__nv_bfloat162*/);
  // End
}
