#include "cuda_bf16.h"

__global__ void test(__nv_bfloat16 b) {
  // Start
  __bfloat162float(b /*__nv_bfloat16*/);
  // End
}
