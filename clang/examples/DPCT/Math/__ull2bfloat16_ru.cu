// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(unsigned long long u) {
  // Start
  __ull2bfloat16_ru(u /*unsigned long long*/);
  // End
}
