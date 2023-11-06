#include "cuda_bf16.h"

__global__ void test(__nv_bfloat162 b1, __nv_bfloat162 b2) {
  // Start
  __lows2bfloat162(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
  // End
}
