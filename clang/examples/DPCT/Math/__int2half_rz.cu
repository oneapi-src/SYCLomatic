// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(int i) {
  // Start
  __int2half_rz(i /*int*/);
  // End
}
