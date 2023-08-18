// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half h1, __half h2) {
  // Start
  hdiv(h1 /*__half*/, h2 /*__half*/);
  // End
}
