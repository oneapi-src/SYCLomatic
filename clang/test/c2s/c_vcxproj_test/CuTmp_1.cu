// UNSUPPORTED: -linux-
// RUN: echo "empty command"

#include "cuda_runtime.h"
#include <stdio.h>

// This is trick skill, used to check CuTmp_1.cu is skipped, for it is in <None> node of compilation database.
// CHECK: __global__ void addKernel(int *c, const int *a, const int *b)
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // CHECK: int i = threadIdx.x;
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
