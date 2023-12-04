#include "cuda_bf16.h"

__global__ void test(float f) {
  // Start
  __float2bfloat162_rn(f /*float*/);
  // End
}
