#include "cuda_fp16.h"

__global__ void test(__half2 h) {
  // Start
  __high2float(h /*__half2*/);
  // End
}
