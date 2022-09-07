// RUN: echo "0"
#include <cuda_runtime.h>

__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
