// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(long long ll) {
  // Start
  __ll2half_rd(ll /*long long*/);
  // End
}
