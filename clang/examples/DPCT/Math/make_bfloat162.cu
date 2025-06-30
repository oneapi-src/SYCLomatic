#include "cuda_bf16.h"

__global__ void test(__nv_bfloat16 bf1, __nv_bfloat16 bf2) {
  // Start
  make_bfloat162(bf1 /*__nv_bfloat16*/, bf2 /*__nv_bfloat16*/);
  // End
}
