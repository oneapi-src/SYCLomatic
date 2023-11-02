// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(short s) {
  // Start
  __short2bfloat16_rd(s /*short*/);
  // End
}
