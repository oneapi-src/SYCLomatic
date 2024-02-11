#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *h, __half2 *h2, float *b,
                     float2 *b2) {
  // Start
  /* 1 */ __ldg(h /*__half **/);
  /* 2 */ __ldg(h2 /*__half2 **/);
  /* 3 */ __ldg(b /*__float **/);
  /* 4 */ __ldg(b2 /*__float2 **/);
  // End
}
