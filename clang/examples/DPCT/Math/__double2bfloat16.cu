// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"

__global__ void test(double d) {
  // Start
  __double2bfloat16(d /*double*/);
  // End
}
