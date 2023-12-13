#include "cuda_fp16.h"

__global__ void test(__half2 h) {
  // Start
  __high2half2(h /*__half2*/);
  // End
}
