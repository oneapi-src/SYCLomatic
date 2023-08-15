#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, __nv_bfloat16 *b,
                     __nv_bfloat162 *b2) {
  // Start
  __ldcs(h /*__half **/);
  __ldcs(h2 /*__half2 **/);
  __ldcs(b /*__nv_bfloat16 **/);
  __ldcs(b2 /*__nv_bfloat162 **/);
  // End
}
