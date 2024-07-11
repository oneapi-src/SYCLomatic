#include "cuda_bf16.h"

__global__ void test(float f1, float f2) {
  // Start
  __floats2bfloat162_rn(f1 /*float*/, f2 /*float*/);
  // End
}
