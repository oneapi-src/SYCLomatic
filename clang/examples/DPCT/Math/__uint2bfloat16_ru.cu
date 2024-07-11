// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(unsigned u) {
  // Start
  __uint2bfloat16_ru(u /*unsigned*/);
  // End
}
