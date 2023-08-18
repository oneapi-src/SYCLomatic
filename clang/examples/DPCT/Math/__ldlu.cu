#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, __nv_bfloat16 *b,
                     __nv_bfloat162 *b2) {
  // Start
  __ldlu(h /*__half **/);
  __ldlu(h2 /*__half2 **/);
  __ldlu(b /*__nv_bfloat16 **/);
  __ldlu(b2 /*__nv_bfloat162 **/);
  // End
}
