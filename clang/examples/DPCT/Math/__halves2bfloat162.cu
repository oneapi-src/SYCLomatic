#include "cuda_bf16.h"

__global__ void test(__nv_bfloat16 b1, __nv_bfloat16 b2) {
  // Start
  __halves2bfloat162(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
  // End
}
