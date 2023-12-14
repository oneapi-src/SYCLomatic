// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(unsigned u) {
  // Start
  __uint2half_rz(u /*unsigned*/);
  // End
}
