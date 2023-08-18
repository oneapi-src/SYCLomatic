// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h1, __half2 h2) {
  // Start
  h2div(h1 /*__half2*/, h2 /*__half2*/);
  // End
}
