// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_fp16.h"

__global__ void test(__half h) {
  // Start
  __half2ll_rz(h /*__half*/);
  // End
}
