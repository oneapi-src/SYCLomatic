// RUN: c2s --format-range=none -out-root %T/asm %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/asm.dp.cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void gpu_ptx(int *d_ptr, int length) {
  int elemID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int innerloops = 0; innerloops < 100000; innerloops++) {
    if (elemID < length) {
      unsigned int laneid;
      // CHECK: /*
      // CHECK-NEXT: DPCT1053:0: Migration of device assembly code is not supported.
      // CHECK-NEXT: */
      asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
      d_ptr[elemID] = laneid;
    }
  }
}

int main(int argc, char **argv) {
  const int N = 1000;
  int *d_ptr;
  cudaMalloc(&d_ptr, N * sizeof(int));
  int *h_ptr;
  cudaMallocHost(&h_ptr, N * sizeof(int));

  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 cudaBlockSize(256, 1, 1);
  dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
  gpu_ptx<<<cudaGridSize, cudaBlockSize >>>(d_ptr, N);
  cudaGetLastError();
  cudaDeviceSynchronize();


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  printf("Time Used on GPU:%f(ms)\n", time_elapsed);

  __asm__("movl %esp,%eax");

  return 0;
  // CHECK: printf("Time Used on GPU:%f(ms)\n", time_elapsed);
  // CHECK-NOT: DPCT1053:0: Migration of device assembly code is not supported.
  // CHECK: return 0;
}

