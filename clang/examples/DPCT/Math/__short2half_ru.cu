// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(short s) {
  // Start
  __short2half_ru(s /*short*/);
  // End
}
