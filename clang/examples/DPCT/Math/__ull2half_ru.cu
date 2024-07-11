// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(unsigned long long u) {
  // Start
  __ull2half_ru(u /*unsigned long long*/);
  // End
}
