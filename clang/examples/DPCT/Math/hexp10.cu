// Option: --use-dpcpp-extensions=intel_device_math
// Option: --use-experimental-features=bfloat16_math_functions
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half h, __nv_bfloat16 b) {
  // Start
  hexp10(h /*__half*/);
  hexp10(b /*__nv_bfloat16*/);
  // End
}
