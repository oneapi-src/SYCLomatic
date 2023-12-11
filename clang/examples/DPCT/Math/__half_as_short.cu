#include "cuda_fp16.h"

__global__ void test(__half h) {
  // Start
  __half_as_short(h /*__half*/);
  // End
}
