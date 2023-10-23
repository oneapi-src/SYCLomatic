#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h1, __half2 h2, __nv_bfloat162 b1,
                     __nv_bfloat162 b2) {
  // Start
  __hgtu2_mask(h1 /*__half2*/, h2 /*__half2*/);
  __hgtu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
  // End
}
