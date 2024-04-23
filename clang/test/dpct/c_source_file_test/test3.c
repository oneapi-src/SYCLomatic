// RUN: echo "empty command"
#include <cuda_runtime.h>

void bar() {
    int *a;
    cudaMalloc(&a, 10);
}