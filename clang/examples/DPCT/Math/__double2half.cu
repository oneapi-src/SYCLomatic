#include "cuda_fp16.h"

__global__ void test(__half h) {
  // Start
  __double2half(h /*__half*/);
  // End
}
