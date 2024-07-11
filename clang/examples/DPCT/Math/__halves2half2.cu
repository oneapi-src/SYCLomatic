#include "cuda_fp16.h"

__global__ void test(__half h1, __half h2) {
  // Start
  __halves2half2(h1 /*__half*/, h2 /*__half*/);
  // End
}
