// RUN: dpct --format-range=none -out-root %T/asm %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/asm.dp.cpp
// clang-format off
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void gpu_ptx(int *d_ptr, int length) {
  int elemID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int innerloops = 0; innerloops < 100000; innerloops++) {
    if (elemID < length) {
      unsigned int laneid;
      // CHECK: /*
      // CHECK-NEXT: DPCT1053:{{[0-9]+}}: Migration of device assembly code is not supported.
      // CHECK-NEXT: */
      asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
      d_ptr[elemID] = laneid;
    }
  }
}

// CHECK:void mov(float *output) {
// CHECK-NEXT: unsigned p;
// CHECK-NEXT: double d;
// CHECK-NEXT: p = 123 * 123U + 456 * ((4 ^ 7) + 2 ^ 3) | 777 & 128 == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;
// CHECK: (*output) = [](){union {uint32_t I; float F;}; I = 0x3f800000u; return F;}();
// CHECK: (*output) = [](){union {uint32_t I; float F;}; I = 0x3f800000u; return F;}();
// CHECK: d = [](){union {uint64_t I; double F;}; I = 0x40091EB851EB851Fu; return F;}();
// CHECK: d = [](){union {uint64_t I; double F;}; I = 0x40091EB851EB851Fu; return F;}();
// CHECK: *output = p;
// CHECK-NEXT:}
__global__ void mov(float *output) {
  unsigned p;
  double d;
  asm ("mov.s32 %0, 123 * 123U + 456 * ((4 ^7) + 2 ^ 3) | 777 & 128 == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;" : "=r"(p) );
  asm ("mov.s32 %0, 0F3f800000;" : "=r"(*output));
  asm ("mov.s32 %0, 0f3f800000;" : "=r"(*output));
  asm ("mov.s32 %0, 0D40091EB851EB851F;" : "=r"(d));
  asm ("mov.s32 %0, 0d40091EB851EB851F;" : "=r"(d));
  *output = p;
}

// CHECK: int cond(int x) {
// CHECK-NEXT: int y = 0;
// CHECK-NEXT: {
// CHECK-NEXT: bool p;
// CHECK-NEXT: uint32_t r[10];
// CHECK-NEXT: r[1] = 0x3F;
// CHECK-NEXT: r[2] = 2023;
// CHECK-NEXT: p = x == 34;
// CHECK-NEXT: if (p) {
// CHECK-NEXT: y = 1;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: return y;
// CHECK-NEXT:}
__device__ int cond(int x) {
  int y = 0;
  asm("{\n\t"
      " .reg .pred %%p;\n\t"
      " .reg .u32 %%r<10>;\n\t"
      " mov.u32 %%r1, 0x3F;\n\t"
      " mov.u32 %%r2, 2023;\n\t"
      " setp.eq.s32 %%p, %1, 34;\n\t" // x == 34?
      " @%%p mov.s32 %0, 1;\n\t"      // set y to 1 if true
      "}"                             // conceptually y = (x==34)?1:y
      : "+r"(y) : "r" (x));
  return y;
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
  // CHECK-NOT: DPCT1053:{{[0-9]+}}: Migration of device assembly code is not supported.
  // CHECK: return 0;
}
// clang-format on
