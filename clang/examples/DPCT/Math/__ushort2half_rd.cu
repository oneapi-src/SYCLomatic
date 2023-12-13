// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(unsigned short u) {
  // Start
  __ushort2half_rd(u /*unsigned short*/);
  // End
}
