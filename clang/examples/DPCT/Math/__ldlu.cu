#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, __nv_bfloat16 *b,
                     __nv_bfloat162 *b2) {
  // Start
  /* 1 */ __ldlu(h /*__half **/);
  /* 2 */ __ldlu(h2 /*__half2 **/);
  /* 3 */ __ldlu(b /*__nv_bfloat16 **/);
  /* 4 */ __ldlu(b2 /*__nv_bfloat162 **/);
  // End
}
