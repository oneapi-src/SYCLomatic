// Option: --use-dpcpp-extensions=intel_device_math
// Option: --use-experimental-features=bfloat16_math_functions
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h1, __half2 h2, __half2 h3, __nv_bfloat162 b1,
                     __nv_bfloat162 b2, __nv_bfloat162 b3) {
  // Start
  __hfma2_relu(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
  __hfma2_relu(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/,
               b3 /*__nv_bfloat162*/);
  // End
}
