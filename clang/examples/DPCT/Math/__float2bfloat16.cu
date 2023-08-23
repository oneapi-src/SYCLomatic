#include "cuda_bf16.h"

__global__ void test(float f) {
  // Start
  __float2bfloat16(f /*float*/);
  // End
}
