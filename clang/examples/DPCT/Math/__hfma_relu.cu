// Option: --use-dpcpp-extensions=intel_device_math
// Option: --use-experimental-features=bfloat16_math_functions
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half h1, __half h2, __half h3, __nv_bfloat16 b1,
                     __nv_bfloat16 b2, __nv_bfloat16 b3) {
  // Start
  __hfma_relu(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
  __hfma_relu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
  // End
}
