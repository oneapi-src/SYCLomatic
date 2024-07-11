#include "cuda_fp16.h"

__global__ void test(__half h) {
  // Start
  __half2half2(h /*__half*/);
  // End
}
