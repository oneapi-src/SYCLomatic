// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h1, __half2 h2, __half2 h3, __nv_bfloat162 b1,
                     __nv_bfloat162 b2, __nv_bfloat162 b3) {
  // Start
  __hcmadd(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
  __hcmadd(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/, b3 /*__nv_bfloat162*/);
  // End
}
