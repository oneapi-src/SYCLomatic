#include "cuda_fp16.h"
__global__ void test(__half h1, __half h2, __half h3) { __hfma_sat(h1, h2, h3); }
