// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(unsigned short u) {
  // Start
  __ushort2bfloat16_rn(u /*unsigned short*/);
  // End
}
