// RUN: echo "empty command"

#include "test.cuh"
#include "test.h"

__device__ int z = 0;

__global__ void f() { y = x; }
