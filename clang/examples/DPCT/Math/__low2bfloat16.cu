#include "cuda_bf16.h"

__global__ void test(__nv_bfloat162 b) {
  // Start
  __low2bfloat16(b /*__nv_bfloat162*/);
  // End
}
