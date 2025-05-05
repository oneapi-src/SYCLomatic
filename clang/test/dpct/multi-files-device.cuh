// RUN: echo "empty command"

#include "multi-files-kernel.cuh"

// CHECK: int global_id() {
__device__ int global_id() {
  unsigned x = 0;
  ATOMIC_UPDATE(x)
  return blockIdx.x * blockDim.x + threadIdx.x;
}

void foo(){
  sgemm();
  randomGen();
}
