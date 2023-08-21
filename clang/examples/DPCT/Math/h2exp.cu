// Option: --use-dpcpp-extensions=intel_device_math
// Option: --use-experimental-features=bfloat16_math_functions
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h, __nv_bfloat162 b) {
  // Start
  h2exp(h /*__half2*/);
  h2exp(b /*__nv_bfloat162*/);
  // End
}
