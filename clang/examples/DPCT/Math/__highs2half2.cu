#include "cuda_fp16.h"

__global__ void test(__half2 h1, __half2 h2) {
  // Start
  __highs2half2(h1 /*__half2*/, h2 /*__half2*/);
  // End
}
