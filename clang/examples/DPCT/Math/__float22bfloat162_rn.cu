#include "cuda_bf16.h"

__global__ void test(float2 f) {
  // Start
  __float22bfloat162_rn(f /*float2*/);
  // End
}
