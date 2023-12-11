#include "cuda_fp16.h"

__global__ void test(short s) {
  // Start
  __short_as_half(s /*short*/);
  // End
}
