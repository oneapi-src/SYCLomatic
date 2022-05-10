// RUN: echo "empty command"
#include "kernel.cuh"

__device__ int a = 0;

__global__ void kernel(){
  a = 1;
}
