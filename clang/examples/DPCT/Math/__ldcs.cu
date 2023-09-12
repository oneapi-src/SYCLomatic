#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, __nv_bfloat16 *b,
                     __nv_bfloat162 *b2) {
  // Start
  /* 1 */ __ldcs(h /*__half **/);
  /* 2 */ __ldcs(h2 /*__half2 **/);
  /* 3 */ __ldcs(b /*__nv_bfloat16 **/);
  /* 4 */ __ldcs(b2 /*__nv_bfloat162 **/);
  // End
}
