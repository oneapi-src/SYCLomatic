#include "cuda_fp16.h"

__global__ void test(unsigned short u) {
  // Start
  __ushort_as_half(u /*unsigned short*/);
  // End
}
