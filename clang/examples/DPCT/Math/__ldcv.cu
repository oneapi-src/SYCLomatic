#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, __nv_bfloat16 *b,
                     __nv_bfloat162 *b2) {
  // Start
  __ldcv(h /*__half **/);
  __ldcv(h2 /*__half2 **/);
  __ldcv(b /*__nv_bfloat16 **/);
  __ldcv(b2 /*__nv_bfloat162 **/);
  // End
}
