// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(float f1, float f2) {
  // Start
  __floats2half2_rn(f1 /*float*/, f2 /*float*/);
  // End
}
